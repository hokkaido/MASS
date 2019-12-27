import numpy as np
import torch
import random
import time
import math

from .ner import align_tokens, ENTITY_TYPES

from fairseq import utils
from fairseq.data import data_utils, BaseWrapperDataset, Dictionary

import spacy

# sequence = size T
def create_segments(sequence, split_idx, max_segments):
    segments = torch.zeros_like(sequence)
    
    for segment_token in split_idx:
        segments += sequence.eq(segment_token).type_as(segments)
        
    segments = torch.cumsum(segments, dim=0) + 1

    return segments.clamp(0, max_segments)

def collate_src_map(data, left_pad=False):
    src_size = max([t.size(0) for t in data])
    src_vocab_size = max([t.max() for t in data]) + 1
    alignment = torch.zeros(len(data), src_size, src_vocab_size).bool()
    for i, sent in enumerate(data):
        for j, t in enumerate(sent):
            if left_pad:
                alignment[i, src_size - len(sent) + j, t] = 1
            else:
                alignment[i, j, t] = 1
    return alignment

def collate_copy_alignment(data):
    tgt_size = max([t.size(0) for t in data])
    alignment = data[0].new(len(data), tgt_size).fill_(0)
    for i, sent in enumerate(data):
        alignment[i,:sent.size(0)] = sent
    return alignment

def dynamic_dict(sample, src_dict):
    """Create copy-vocab and numericalize with it.
    In-place adds ``"src_map"`` to ``example``. That is the copy-vocab
    numericalization of the tokenized ``example["src"]``. If ``example``
    has a ``"tgt"`` key, adds ``"alignment"`` to example. That is the
    copy-vocab numericalization of the tokenized ``example["tgt"]``. The
    alignment has an initial and final UNK token to match the BOS and EOS
    tokens.
    Args:
        example (dict): An example dictionary with a ``"src"`` key and
            maybe a ``"tgt"`` key. (This argument changes in place!)
        src_field (torchtext.data.Field): Field object.
        tgt_field (torchtext.data.Field): Field object.
    Returns:
        torchtext.data.Vocab and ``example``, changed as described.
    """

    src = sample['source']

    # make a small vocab containing just the tokens in the source sequence
    src_ex_dict = Dictionary(unk=src_dict.unk_word, pad=src_dict.pad_word)

    for idx in src:
        src_ex_dict.add_symbol(src_dict[idx])

    # Map source tokens to indices in the dynamic dict.
    src_map = src.new([src_ex_dict.index(src_dict[idx]) for idx in src])

    sample["src_map"] = src_map
    sample["src_ex_dict"] = src_ex_dict

    if "target" in sample:
        tgt = sample["target"]
        mask = src.new([src_ex_dict.index(src_dict[idx]) for idx in tgt])
        sample['copy_alignment'] = mask

def collate(
    samples, src_dict, ent_pad_idx, ent_eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True, copy_attn=False, max_segments=0
):
    if len(samples) == 0:
        return {}

    pad_idx=src_dict.pad()
    eos_idx=src_dict.eos()

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def merge_seg(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            0, max_segments + 1, left_pad, move_eos_to_beginning,
        )

    def merge_ent(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            ent_pad_idx, ent_eos_idx, left_pad, move_eos_to_beginning,
        )

    def prepare_copy_attn():
        for s in samples:
            dynamic_dict(s, src_dict)

    id = torch.LongTensor([s['id'] for s in samples])

    if copy_attn:
        prepare_copy_attn()

    src_tokens = merge('source', left_pad=left_pad_source)
    src_entities = merge_ent('source_entities', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    src_entities = src_entities.index_select(0, sort_order)
    src_segments = None
    prev_output_tokens = None
    prev_output_entities = None
    prev_output_segments = None
    src_map = None
    src_ex_dict = None
    alignment = None

    if copy_attn:
        src_ex_dict = [samples[idx]['src_ex_dict'] for idx in sort_order]
        # pad will always be 0 with src dicts
        src_map = collate_src_map([s['src_map'] for s in samples], left_pad=left_pad_source)
        src_map = src_map.index_select(0, sort_order)

    if samples[0].get('source_segments', None) is not None:
        src_segments = merge_seg('source_segments', left_pad=left_pad_source)
        src_segments = src_segments.index_select(0, sort_order)

    if samples[0].get('target_segments', None) is not None:
        prev_output_segments = merge_seg('target_segments', left_pad=left_pad_target)
        prev_output_segments = prev_output_segments.index_select(0, sort_order)

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
            prev_output_entities = merge_ent(
                'target_entities',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_entities = prev_output_entities.index_select(0, sort_order)
        if copy_attn:
            alignment = merge('copy_alignment', left_pad=left_pad_target)
            alignment = alignment.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_entities': src_entities,
            'src_segments': src_segments,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    if prev_output_entities is not None:
        batch['net_input']['prev_output_entities'] = prev_output_entities
    if prev_output_segments is not None:
        batch['net_input']['prev_output_segments'] = prev_output_segments
    if src_ex_dict is not None:
        batch['src_ex_dict'] = src_ex_dict
    if src_map is not None:
        batch['src_map'] = src_map
    if alignment is not None:
        batch['copy_alignment'] = alignment

    return batch

def truncate(data, max_len, eos_idx):
    has_eos = data[-1] == eos_idx
    data = data[:max_len]
    if has_eos: 
        data[-1] = eos_idx
    else:
        print('NO WE DONT HAVE')
    return data

class AugmentedLanguagePairDataset(BaseWrapperDataset):
    """ Wrapper for augmented language pair datasets 
        
        Requires a shared vocabulary
    """
    def __init__(self, 
        dataset, 
        entities, 
        copy_attn=False, 
        segment_tokens=None, 
        max_segments=None, 
        truncate_source_positions=None, 
        truncate_target_positions=None):
        super().__init__(dataset)
        self.entities = entities
        self.copy_attn = copy_attn
        self.segment_tokens = segment_tokens
        self.max_segments = max_segments
        self.truncate_source_positions = truncate_source_positions
        self.truncate_target_positions = truncate_target_positions
        self.src_segment_tokens_idx = None
        self.tgt_segment_tokens_idx = None

        if self.segment_tokens is not None and self.max_segments is not None:
            segment_tokens = self.segment_tokens.split(',')
            self.src_segment_tokens_idx = [self.dataset.src_dict.index(token) for token in segment_tokens]
            self.tgt_segment_tokens_idx = [self.dataset.tgt_dict.index(token) for token in segment_tokens]

    def __getitem__(self, index):

        sample = self.dataset[index]
        entities = self.entities[index]

        sample['source_entities'] = entities['source']
        sample['target_entities'] = entities['target']

        if self.src_segment_tokens_idx is not None:
            sample['source_segments'] = create_segments(
                sample['source'], 
                self.src_segment_tokens_idx, 
                self.max_segments)
                
        if self.tgt_segment_tokens_idx is not None:
            sample['target_segments'] = create_segments(
                sample['target'], 
                self.tgt_segment_tokens_idx, 
                self.max_segments)

        if self.truncate_source_positions is not None:
            for key, eos_idx in zip(
                ['source', 'source_entities', 'source_segments'], 
                [self.dataset.src_dict.eos(), self.entities.src_dict.eos(), self.max_segments + 1]):
                if key in sample and len(sample[key]) > self.truncate_source_positions:
                    sample[key] = truncate(sample[key], self.truncate_source_positions, eos_idx)

        if self.truncate_target_positions is not None:
            for key, eos_idx in zip(
                ['targeta', 'target_entities', 'target_segments'], 
                [self.dataset.tgt_dict.eos(), self.entities.tgt_dict.eos(), self.max_segments + 1]):
                if key in sample and len(sample[key]) > self.truncate_target_positions:
                    sample[key] = truncate(sample[key], self.truncate_target_positions, eos_idx)

        return sample
        
    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch with the following keys:
                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:
                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, src_dict=self.dataset.src_dict,
            ent_pad_idx=self.entities.src_dict.pad(), ent_eos_idx=self.entities.src_dict.eos(),
            left_pad_source=self.dataset.left_pad_source, left_pad_target=self.dataset.left_pad_target,
            input_feeding=self.dataset.input_feeding, copy_attn=self.copy_attn, max_segments=self.max_segments
        )

