import numpy as np
import torch
import random
import time
import math

from fairseq import utils
from fairseq.data import data_utils, BaseWrapperDataset

class MaskedSummarizationDataset(BaseWrapperDataset):
    """ Wrapper for masked summarization datasets 
        
        Requires a shared vocabulary

        
        [x1, x2, x3, x4, x5] => [y1, y2, y3]
                 ||
                 VV
        [x1,  _,  _, x4, x5] => [y1, y2, y3]

        default,  _ will be replaced by 8:1:1 (mask, self, rand),
    """
    def __init__(self, dataset, pred_probs=None):
        super().__init__(dataset)      
        self.pred_probs = pred_probs
        
    def __getitem__(self, index):

        item = self.dataset[index]

        tgt_item = item['target']
        src_item = item['source']
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa       

        shared, masked_src_idx, masked_tgt_idx = np.intersect1d(src_item, tgt_item, return_indices=True)

        src_item[masked_src_idx] = self.replace(src_item[masked_src_idx])
        
        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

    def replace(self, x):
        _x_real = x
        _x_rand = _x_real.clone().random_(self.dataset.src_dict.nspecial, len(self.dataset.src_dict))
        _x_mask = _x_real.clone().fill_(self.dataset.src_dict.index('[MASK]'))
        probs = torch.multinomial(self.pred_probs, len(x), replacement=True)
        _x = _x_mask * (probs == 0).long() + \
             _x_real * (probs == 1).long() + \
             _x_rand * (probs == 2).long()
        return _x
