import numpy as np
import torch
import random
import time
import math

from .ner import align_tokens, ENTITY_TYPES

from fairseq import utils
from fairseq.data import data_utils, BaseWrapperDataset

import spacy

nlp = spacy.load("spacy_models/en_core_web_sm_lower")

# sequence = B x T
def create_segments(sequence, segment_tokens, max_segments):
    bsz, T = sequence.shape
    segments = torch.zeros_like(sequence)
    
    for segment_token in segment_tokens:
        segments += sequence.eq(segment_token).type_as(segments)
        
    segments = torch.cumsum(segments, dim=1)
    narrowed = torch.narrow(segments, 1, 1, T - 1)
    append = torch.zeros((bsz, 1)).type_as(narrowed)

    return torch.cat((append, narrowed), 1).clamp(0, max_segments - 1)

def create_ner(tokens, src_dict):
    """ Given a 1d Tensor (with token indices), return a list of NER tags
    
    """
    entities = torch.zeros_like(tokens)
    token_list = [src_dict[idx] for idx in tokens]
    pad_idx = src_dict.pad()
    txt = ' '.join(token_list)
    txt = data_utils.process_bpe_symbol(txt, ' ##')
    doc = nlp(txt)

    _, alignments = align_tokens(doc, token_list)
    
    for i in range(len(alignments)):
        spacy_token = doc[i]
        ent_type = pad_idx + 1
        if spacy_token.ent_type_ in ENTITY_TYPES:
            ent_type += ENTITY_TYPES[spacy_token.ent_type_]

        for wp_idx in alignments[i]:
            entities[wp_idx] = ent_type
    
    return entities

def collate_ner_tokens(values, pad_idx, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            dst[0] = pad_idx + 1 # DANGER, was EOS initially, now set to 1, which is equal to no entity type
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

def collate(
    samples, src_dict, left_pad_source=True, left_pad_target=False,
    input_feeding=True, embed_entities=False
):

    pad_idx = src_dict.pad()
    eos_idx = src_dict.eos()

    if len(samples) == 0:
        return {}

    def collate_token_types():
        for s in samples:
            s['source_entities'] = create_ner(s['source'], src_dict)
            s['target_entities'] = create_ner(s['target'], src_dict)

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def merge_ner(key, left_pad, move_eos_to_beginning=False):
        return collate_ner_tokens(
            [s[key] for s in samples],
            pad_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
          
    src_tokens = merge('source', left_pad=left_pad_source)
    src_entities = None

    if embed_entities:
        collate_token_types()
        src_entities = merge_ner('source_entities', left_pad=left_pad_source)

    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    if embed_entities:
        src_entities = src_entities.index_select(0, sort_order)
    prev_output_tokens = None
    prev_output_entities = None
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
            if embed_entities:
                prev_output_entities = merge_ner(
                    'target_entities',
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                )
                prev_output_entities = prev_output_entities.index_select(0, sort_order)
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
    assert src_tokens.numel() == src_entities.numel()

    if src_entities is not None:
        batch['net_input']['src_entities'] = src_entities

    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    if prev_output_entities is not None:
        assert prev_output_tokens.numel() == prev_output_entities.numel()
        batch['net_input']['prev_output_entities'] = prev_output_entities

    return batch

class SegmentedLanguagePairDataset(BaseWrapperDataset):
    """ Wrapper for segmented language pair datasets 
        
        Requires a shared vocabulary
    """
    def __init__(self, dataset, embed_entities=False, segment_tokens_idx=None, max_segments=None):
        super().__init__(dataset)      
        self.embed_entities = embed_entities
        self.segment_tokens_idx = segment_tokens_idx
        self.max_segments = max_segments
        
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
        batch = collate(
            samples, self.dataset.src_dict,
            left_pad_source=self.dataset.left_pad_source, left_pad_target=self.dataset.left_pad_target,
            input_feeding=self.dataset.input_feeding, embed_entities=self.embed_entities
        )
 
        if self.segment_tokens_idx is not None:
            src_item = batch['net_input']['src_tokens']
            tgt_item = batch['net_input']['prev_output_tokens']
            batch['net_input']['src_segment_labels'] = create_segments(src_item, self.segment_tokens_idx, self.max_segments)
            batch['net_input']['tgt_segment_labels'] = create_segments(tgt_item, self.segment_tokens_idx, self.max_segments)
        return batch
