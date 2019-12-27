#from fairseq.data import BertDictionary

import torch
import itertools
import os
import numpy as np

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
)

from .bert_dictionary import BertDictionary
from .masked_summarization_dataset import MaskedSummarizationDataset

@register_task('masked_summarization_mass')
class MaskedSummarizationTask(TranslationTask):
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
        
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict, ent_src_dict, ent_tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.src_dicts = {}
        self.tgt_dicts = {}
        self.src_dicts['core'] = src_dict
        self.src_dicts['entities'] = ent_src_dict
        self.tgt_dicts['core'] = tgt_dict
        self.tgt_dicts['entities'] = ent_tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(os.path.join(paths[0], 'core'))
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'core', 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'core', 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] [{}] dictionary: {} types'.format(args.source_lang, 'core', len(src_dict)))
        print('| [{}] [{}] dictionary: {} types'.format(args.target_lang, 'core', len(tgt_dict)))

        ent_src_dict = cls.load_dictionary(os.path.join(paths[0], 'entities', 'dict.{}.txt'.format(args.source_lang)))
        ent_tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'entities', 'dict.{}.txt'.format(args.target_lang)))
        assert ent_src_dict.pad() == ent_tgt_dict.pad()
        assert ent_src_dict.eos() == ent_tgt_dict.eos()
        assert ent_src_dict.unk() == ent_tgt_dict.unk()
        print('| [{}] [{}] dictionary: {} types'.format(args.source_lang, 'entities', len(ent_src_dict)))
        print('| [{}] [{}] dictionary: {} types'.format(args.target_lang, 'entities', len(ent_tgt_dict)))

        return cls(args, src_dict, tgt_dict, ent_src_dict, ent_tgt_dict)

    @classmethod
    def load_dictionary(cls, filename):
        return BertDictionary.load_from_file(filename)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        core = self._load_dataset('core', split, epoch, combine, **kwargs)
        entities = self._load_dataset('entities', split, epoch, combine, **kwargs)

        self.datasets[split] = MaskedSummarizationDataset(
            core,
            entities,
            self.args)

    def _load_dataset(self, ds_name, split, epoch=0, combine=False, **kwargs):
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)] 
        data_path = os.path.join(data_path, ds_name)

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        return load_langpair_dataset(
            data_path, split, src, self.src_dicts[ds_name], tgt, self.tgt_dicts[ds_name],
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return self.args.max_source_positions, self.args.max_target_positions
