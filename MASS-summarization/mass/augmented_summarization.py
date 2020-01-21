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
from .augmented_dataset import AugmentedLanguagePairDataset

@register_task('augmented_summarization_mass')
class AugmentedSummarizationTask(TranslationTask):
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
        parser.add_argument('--embed-segments-encoder', action='store_true',
                            help='Add an additional sentence embedding layer for encoding')
        parser.add_argument('--embed-segments-decoder', action='store_true',
                            help='Add an additional sentence embedding layer for decoding')   
        parser.add_argument('--segment-tokens', default=None, type=str,
                            help='Tokens to use as a segment, i.e. ".,!,?,[SEP]". This will add an additional segment embedding')
        parser.add_argument('--max-segments', default=64, type=int, metavar='N',
                            help='max number of segments to embed in a sequence')
        parser.add_argument('--embed-entities-encoder', action='store_true',
                            help='Add an additional NER embedding layer for encoding')
        parser.add_argument('--embed-entities-decoder', action='store_true',
                            help='Add an additional NER embedding layer for decoding')
        parser.add_argument('--copy-attn', default=False, action='store_true',
                            help='Train copy attention layer.')
        parser.add_argument('--truncate-source-positions', default=None, type=int, metavar='N',
                            help='truncate number of tokens in the source sequence during training and validation')
        parser.add_argument('--truncate-target-positions', default=None, type=int, metavar='N',
                            help='truncate number of tokens in the target sequence during training and validation')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict, ent_src_dict, ent_tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.src_dicts = {}
        self.tgt_dicts = {}
        self.src_dicts['core'] = src_dict
        self.src_dicts['entities'] = ent_src_dict
        self.tgt_dicts['core'] = tgt_dict
        self.tgt_dicts['entities'] = ent_tgt_dict
        self.copy_attn = args.copy_attn
        self.embed_segments_encoder = args.embed_segments_encoder
        self.embed_segments_decoder = args.embed_segments_decoder
        self.segment_tokens = args.segment_tokens
        self.max_segments = args.max_segments
        self.truncate_source_positions = args.truncate_source_positions
        self.truncate_target_positions = args.truncate_target_positions
        self.load_dataset('test')
        print(args)

    @classmethod
    def load_dictionary(cls, filename):
        return BertDictionary.load_from_file(filename)

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

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        core = self._load_dataset('core', split, epoch, combine, **kwargs)
        entities = self._load_dataset('entities', split, epoch, combine, **kwargs)
        truncate_source_positions = self.truncate_source_positions if split != 'test' else None
        truncate_target_positions = self.truncate_target_positions if split != 'test' else None
        self.datasets[split] = AugmentedLanguagePairDataset(
            core,
            entities,
            copy_attn=self.copy_attn,
            segment_tokens=self.segment_tokens,
            max_segments=self.max_segments,
            truncate_source_positions=truncate_source_positions,
            truncate_target_positions=truncate_target_positions)

        #ds = self.datasets[split]
        
        # print(split)
        # src_max = 0
        # tgt_max = 0

        # for idx in range(len(ds)):
        #     item = ds[idx]
        #     sm = len(item['source'])
        #     tm = len(item['target'])
        #     if sm > src_max:
        #         src_max = sm
        #         print(' '.join([ds.dataset.src_dict[t] for t in item['source']]))
        #         print('new src max ', split, ' ', src_max)

        #     if tm > tgt_max:
        #         tgt_max = tm
        #         print(' '.join([ds.dataset.tgt_dict[t] for t in item['target']]))
        #         print('new tgt max ', split, ' ', tgt_max)
    
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

    def build_dataset_for_inference(self, src_tokens, src_lengths):
       print('woot')
       return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.
        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True
        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def build_generator(self, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            from fairseq.sequence_generator import SequenceGenerator
            from .copy_sequence_generator import CopySequenceGenerator
            if getattr(args, 'copy_attn', False):
                seq_gen_cls = CopySequenceGenerator
            else:
                seq_gen_cls = SequenceGenerator
            return seq_gen_cls(
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                sampling=getattr(args, 'sampling', False),
                sampling_topk=getattr(args, 'sampling_topk', -1),
                sampling_topp=getattr(args, 'sampling_topp', -1.0),
                temperature=getattr(args, 'temperature', 1.),
                diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return self.args.max_source_positions, self.args.max_target_positions