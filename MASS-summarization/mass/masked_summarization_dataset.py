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

    def merge(x, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            x, pad_idx, eos_idx, left_pad, move_eos_to_beginning
        )
    
    id = torch.LongTensor([s['id'] for s in samples])
    source = merge([s['source'] for s in samples], left_pad=left_pad_source)
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    prev_output_tokens = merge([s['prev_output_tokens'] for s in samples], left_pad=left_pad_target)
    positions = merge([s['positions'] for s in samples], left_pad=left_pad_target)
    target = merge([s['target'] for s in samples], left_pad=left_pad_target)
    ntokens = target.numel()

    batch = {
        'id' : id,
        'nsentences': len(samples),
        'net_input' : {
            'src_lengths': src_lengths,
            'src_tokens' : source,
            'prev_output_tokens': prev_output_tokens,
            'positions': positions
        },
        'target' : target,
        'ntokens': ntokens,
    }
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
        self.permute_sentence_ratio = 1.0
        self.mask_idx = self.dataset.src_dict.index('[MASK]')
        self.pad_idx = self.dataset.src_dict.pad()
        self.full_stop_idx = self.dataset.src_dict.index('.')
        self.swap_ratio = 0.5
        self.mask_ratio = 0.15
        self.replacement_ratio = 0.01
        
    def __getitem__(self, index):
        sample = self.dataset[index]
        entities = self.entities[index]

        source = sample['source']
        target = sample['target']
        new_target = source.clone()

        assert source[-1] == self.dataset.src_dict.eos()

        if self.permute_sentence_ratio > 0.0:
            source = self.permute_sentences(source, self.permute_sentence_ratio)

        if self.mask_ratio > 0.0:
            source, mask = self.add_overlap_mask(source, target, self.swap_ratio, self.mask_ratio)

        if self.replacement_ratio > 0.0:
            source = self.add_replacement_noise(source, self.replacement_ratio)

        masked_pos = mask.nonzero()[0]
        prev_output_tokens = new_target[masked_pos - 1].clone()
        new_target = new_target[mask].clone()        
        positions = torch.LongTensor(masked_pos) + self.pad_idx

        assert source[-1] == self.dataset.src_dict.eos()

        return {
            'id': index,
            'source': source,
            'target': new_target,
            'prev_output_tokens': prev_output_tokens,
            'positions': positions
        }

    def print_sentence(self, source):
        print(' '.join(self.dataset.src_dict[idx] for idx in source))

    def permute_sentences(self, source, p=1.0):
        if len(source) < 2:
            return source
        full_stops = (source == self.full_stop_idx)
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

        index = 0
        for i in ordering:
            sentence = source[(sentence_ends[i - 1] if i > 0 else 1):sentence_ends[i]]
            result[index:index + sentence.size(0)] = sentence
            index += sentence.size(0)
        return result

    def add_replacement_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens) - 1
        n = int(math.ceil(num_tokens * p))

        replacement = torch.randint(low=self.dataset.src_dict.nspecial, high=len(self.dataset.src_dict), size=(n,))
        rep = torch.randperm(num_tokens)
        noise_indices = rep[:n]

        tokens[noise_indices] = replacement

        assert (tokens >= 0).all()
        return tokens

    def add_overlap_mask(self, source, target, swap_p=0.5, mask_p=0.3):

        num_tokens = len(source) - 1
        mask_budget = int(math.ceil(num_tokens * mask_p))

        source_overlap = np.in1d(source[:-1], target[:-1])
        #target_overlap = np.in1d(target[:-1], source[:-1])

        if random.random() >= swap_p:
            source_overlap = ~source_overlap
            #target_overlap = ~target_overlap

        num_overlaps = source_overlap.sum()
        if num_overlaps > mask_budget:
            source_overlap[source_overlap] = torch.FloatTensor(num_overlaps).uniform_() < (mask_budget / num_overlaps)

            
        source[:-1][source_overlap] = self.mask_idx
        #target[:-1][target_overlap] = target_overlap
        #target = target[np.append(target_overlap, True)]

        return source, np.append(source_overlap, True) # EOS

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return collate(samples, self.dataset.src_dict.pad(), self.dataset.src_dict.eos(), self.dataset.left_pad_source, self.dataset.left_pad_target)