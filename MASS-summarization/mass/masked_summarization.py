#from fairseq.data import BertDictionary

import torch
import itertools
import os
import numpy as np
from collections import OrderedDict

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
    RoundRobinZipDatasets,
)

from .bert_dictionary import BertDictionary
from .masked_summarization_dataset import MaskedSummarizationDataset
from .noisy_language_pair_dataset import NoisyLanguagePairDataset

def load_noisy_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions, max_target_positions,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        print(filename, flush=True)
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_datasets.append(
            data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        )
        tgt_datasets.append(
            data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        )

        print('| {} {} {}-{} {} examples'.format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    print(src_dataset.sizes)
    print(tgt_dataset.sizes)
    return NoisyLanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
    )

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
        parser.add_argument('--word_mask_keep_rand', default="0.1,0.1,0.8", type=str,
                            help='Word prediction proability')
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

        s = args.word_mask_keep_rand.split(',')
        s = [float(x) for x in s]
        setattr(args, 'pred_probs', torch.FloatTensor([s[0], s[1], s[2]]))

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

        masked_dataset = MaskedSummarizationDataset(
            core,
            entities,
            self.args)


        noisy_dataset = self._load_noisy_dataset('core', split, epoch, combine, **kwargs)

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict([
                ('masked', masked_dataset),
                ('noisy', noisy_dataset),
                ('core', core),
            ]),
            eval_key=None
        )

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

    def _load_noisy_dataset(self, ds_name, split, epoch=0, combine=False, **kwargs):
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)] 
        data_path = os.path.join(data_path, ds_name)

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        return load_noisy_langpair_dataset(
            data_path, split, src, self.src_dicts[ds_name], tgt, self.tgt_dicts[ds_name],
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

        def forward_backward(model, samples, logging_output_key, weight=1.0):
            nonlocal agg_loss, agg_sample_size, agg_logging_output
            if samples is None or len(samples) == 0:
                return
            loss, sample_size, logging_output = criterion(model, samples)
            if ignore_grad:
                loss *= 0

            optimizer.backward(loss)
            agg_loss += loss.detach().item()
            agg_sample_size += sample_size
            agg_logging_output[logging_output_key] = logging_output

        forward_backward(model, sample['masked'], 'masked')
        forward_backward(model, sample['noisy'], 'noisy')

        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}


            sample_key = 'core'

            loss, sample_size, logging_output = criterion(model, sample[sample_key])

            agg_loss += loss.data.item()
            agg_sample_size += sample_size
            agg_logging_output[sample_key] = logging_output

        return agg_loss, agg_sample_size, agg_logging_output

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        logging_output_keys = {
            key
            for logging_output in logging_outputs
            for key in logging_output
        }

        agg_logging_outputs = {
            key: criterion.__class__.aggregate_logging_outputs([
                logging_output.get(key, {}) for logging_output in logging_outputs
            ])
            for key in logging_output_keys
        }

        def sum_over_languages(key):
            return sum(logging_output[key] for logging_output in agg_logging_outputs.values())

        # flatten logging outputs
        flat_logging_output = {
            '{}:{}'.format(lang_pair, k): v
            for lang_pair, agg_logging_output in agg_logging_outputs.items()
            for k, v in agg_logging_output.items()
        }
        flat_logging_output['loss'] = sum_over_languages('loss')
        if any('nll_loss' in logging_output for logging_output in agg_logging_outputs.values()):
            flat_logging_output['nll_loss'] = sum_over_languages('nll_loss')
        flat_logging_output['sample_size'] = sum_over_languages('sample_size')
        flat_logging_output['nsentences'] = sum_over_languages('nsentences')
        flat_logging_output['ntokens'] = sum_over_languages('ntokens')

        return flat_logging_output

    def max_positions(self):
        return OrderedDict([
            (key, (self.args.max_source_positions, self.args.max_target_positions))
            for key in next(iter(self.datasets.values())).datasets.keys()
        ])