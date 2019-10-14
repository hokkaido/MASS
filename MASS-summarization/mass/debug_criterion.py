import math
import torch
from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion




@register_criterion('debug_criterion')
class DebugCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        torch.set_printoptions(profile="short")
        torch.set_printoptions(threshold=50)
        print(sample)
        net_output = model(**sample['net_input'])
        print('OMGOMG')
        print(net_output)

        torch.set_printoptions(profile="default")