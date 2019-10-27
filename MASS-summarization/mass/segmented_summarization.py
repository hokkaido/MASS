#from fairseq.data import BertDictionary

import torch
import itertools
import os
import numpy as np

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
)

from .bert_dictionary import BertDictionary
from .segmented_dataset import SegmentedLanguagePairDataset

@register_task('segmented_summarization_mass')
class SegmentedSummarizationTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--segment-tokens', default=None, type=str,
                            help='Tokens to use as a segment, i.e. ".,!,?,[SEP]". This will add an additional segment embedding')
        parser.add_argument('--max-segments', default=64, type=int, metavar='N',
                            help='max number of segments to embed in a sequence')
        parser.add_argument('--embed-entities', action='store_true',
                            help='Add an additional NER embedding layer')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

    @classmethod
    def load_dictionary(cls, filename):
        return BertDictionary.load_from_file(filename)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        
        super().load_dataset(split, epoch=0, combine=False, **kwargs)

        segment_tokens_idx=None
        if self.args.segment_tokens is not None:
            segment_tokens = self.args.segment_tokens.split(',')
            segment_tokens_idx = [self.src_dict.index(token) for token in segment_tokens]

        self.datasets[split] = SegmentedLanguagePairDataset(
            self.datasets[split],
            embed_entities=self.args.embed_entities,
            segment_tokens_idx=segment_tokens_idx, 
            max_segments=self.args.max_segments)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return self.args.max_source_positions, self.args.max_target_positions
