import numpy as np
import torch
import random
import time
import math

from fairseq import utils
from fairseq.data import data_utils, BaseWrapperDataset
from .augmented_dataset import AugmentedLanguagePairDataset


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch

class MaskedSummarizationDataset(BaseWrapperDataset):
    """ Wrapper for masked summarization datasets 
        
        Requires a shared vocabulary

        
        [x1, x2, x3, x4, x5] => [y1, y2, y3]
                 ||
                 VV
        [x1,  _,  _, x4, x5] => [y1, y2, y3]

        default,  _ will be replaced by 8:1:1 (mask, self, rand),
    """
    def __init__(self, dataset, entities, args):
        super().__init__(dataset)
        self.entities = entities   
        # self.mask_ratio = args.mask
        # self.random_ratio = args.mask_random
        # self.insert_ratio = args.insert
        # self.rotate_ratio = args.rotate
        # self.permute_sentence_ratio = args.permute_sentences

        self.mask_idx = self.dataset.src_dict.index('[MASK]')
        
    def __getitem__(self, index):
        sample = self.dataset[index]
        entities = self.entities[index]

        source = sample['source']
        target = sample['target']

        assert source[-1] == self.dataset.src_dict.eos()
        assert target[-1] == self.dataset.tgt_dict.eos()

        source_entities = entities['source']
        target_entities = entities['target']

        source_overlap = np.append(np.in1d(source[:-1], target[:-1]), False)
        target_overlap = np.in1d(target, source)
        source[source_overlap] = self.mask_idx
        target = target[target_overlap]

        # if self.permute_sentence_ratio > 0.0:
        #     source = self.permute_sentences(source, self.permute_sentence_ratio)

        # if self.mask_ratio > 0:
        #     source = self.add_whole_word_mask(source, self.mask_ratio)

        # if self.insert_ratio > 0:
        #     source = self.add_insertion_noise(source, self.insert_ratio)

        assert source[-1] == self.dataset.src_dict.eos()

        assert target[-1] == self.dataset.tgt_dict.eos()

        return {
            'id': index,
            'source': source,
            'target': target,
        }

    def permute_sentences(self, source, p=1.0):
        full_stops = (source == self.full_stop_index)
        # Pretend it ends with a full stop so last span is a sentence
        full_stops[-2] = 1

        # Tokens that are full stops, where the previous token is not
        sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero() + 2
        result = source.clone()

        num_sentences = sentence_ends.size(0)
        num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
        substitutions = torch.randperm(num_sentences)[:num_to_permute]
        ordering = torch.arange(0, num_sentences)
        ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

        # Ignore <bos> at start
        index = 1
        for i in ordering:
            sentence = source[(sentence_ends[i - 1] if i > 0 else 1):sentence_ends[i]]
            result[index:index + sentence.size(0)] = sentence
            index += sentence.size(0)
        return result

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(low=1, high=len(self.vocab), size=(num_random,))

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return collate(samples, self.dataset.src_dict.pad(), self.dataset.src_dict.eos(), self.dataset.left_pad_source, self.dataset.left_pad_target)