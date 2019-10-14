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
from .summarization_dataset import MaskedSummarizationDataset

@register_task('summarization_mass')
class SummarizationTask(TranslationTask):
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
        parser.add_argument('--mask-extr-keep-rand', default="0.8,0.1,0.1", type=str,
                            help='Word prediction probability for decoder mask')
        
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

        pred_probs = torch.FloatTensor([float(x) for x in self.args.mask_extr_keep_rand.split(',')])

        self.datasets[split] = MaskedSummarizationDataset(self.datasets[split], pred_probs=pred_probs)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return self.args.max_source_positions, self.args.max_target_positions
