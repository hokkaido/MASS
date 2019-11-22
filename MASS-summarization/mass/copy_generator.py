import math

import torch
import torch.nn as nn

from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion

from .utils import aeq


def collapse_copy_scores(scores, sample, tgt_dict, src_dicts=None,
                         batch_dim=1, batch_offset=None):
    """
    Given scores from an expanded dictionary
    corresponeding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.
    """
    offset = len(tgt_dict)
    for b in range(scores.size(batch_dim)):
        blank = []
        fill = []

        if src_dicts is None:
            src_dict = sample['src_ex_dict'][b]
        else:
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = sample['id'].data[batch_id]
            src_dict = src_dicts[index]

        for i in range(1, len(src_dict)):
            sw = src_dict[i]
            ti = tgt_dict.index(sw)
            if ti != 0:
                blank.append(offset + i)
                fill.append(ti)
        if blank:
            blank = torch.Tensor(blank).type_as(sample['id'].data)
            fill = torch.Tensor(fill).type_as(sample['id'].data)
            score = scores[:, b] if batch_dim == 1 else scores[b]
            score.index_add_(1, fill, score.index_select(1, blank))
            score.index_fill_(1, blank, 1e-10)
    return scores



@register_criterion('copy_generator_loss')
class CopyGeneratorLossCriterion(FairseqCriterion):
    """Copy generator criterion."""
    def __init__(self, args, task):
        super().__init__(args, task)
        self.force_copy = False
        self.eps=1e-5

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # bsz, src_len, max_oov_len
        src_map = sample['src_map']

        # bsz, tgt_len
        align = sample['copy_alignment']

        net_output = model(**sample['net_input'])

        # bsz, tgt_len, src_len
        copy_attn = net_output[1]['copy']

        # bsz, tgt_len
        target = model.get_targets(sample, net_output)     

        scores = model.get_normalized_probs(net_output, log_probs=False, sample=sample)
        target = target.view(-1)
        align = align.view(-1)
        loss = self.compute_loss(model, align, scores, target, reduce)
        sample_size = target.size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, align, scores, target, reduce):
        unk_index = model.decoder.dictionary.unk()
        ignore_index = model.decoder.dictionary.pad()
        vocab_size = len(model.decoder.dictionary)
        # probabilities assigned by the model to the gold targets
        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)

        # probability of tokens copied from source
        copy_ix = align.unsqueeze(1) + vocab_size
        copy_tok_probs = scores.gather(1, copy_ix).squeeze(1)
        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = align == unk_index
        if not self.force_copy:
            non_copy = non_copy | (target != unk_index)

        probs = torch.where(
            non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        )

        loss = -probs.log()  # just NLLLoss; can the module be incorporated?
        # Drop padding.
        loss[target == ignore_index] = 0
        # vocab_size = len(model.decoder.dictionary)
        # vocab_probs = scores.gather(2, target.unsqueeze(2))

        # # probability of tokens copied from source
        # copy_ix = align.unsqueeze(2) + vocab_size

        # copy_tok_probs = scores.gather(2, copy_ix)

        # # Set scores for unk to 0 and add eps
        # copy_tok_probs[align == unk_index] = 0
        # copy_tok_probs += self.eps  # to avoid -inf logs

        # # find the indices in which you do not use the copy mechanism
        # non_copy = align.unsqueeze(2) == unk_index
        # if not self.force_copy:
        #     non_copy = non_copy | (target.unsqueeze(2) != unk_index)

        # probs = torch.where(
        #     non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        # )

        # loss = -probs.squeeze(2).log()  # just NLLLoss; can the module be incorporated?
        # # Drop padding.
        # loss[target == ignore_index] = 0
        if reduce:
            loss = loss.sum()

        return loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }