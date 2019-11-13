import torch
import spacy
from fairseq.data import data_utils

from .segmented_dataset import create_ner
from .ner import align_tokens, ENTITY_TYPES

nlp = spacy.load('spacy_models/en_core_web_sm_lower')

def create_cheap_ner(tokens, src_dict, ent_pad_idx=0, ent_eos_idx=21):
    if tokens.shape[1] == 1:
        # initial inference step, we need to return EOS
        return torch.empty_like(tokens).fill_(ent_eos_idx)

    output_entities = torch.ones_like(tokens)
    output_entities[:, 0] = ent_eos_idx
    return output_entities

def create_ner_from_output_tokens(tokens, src_dict, ent_pad_idx=0, ent_eos_idx=21):

    #if tokens.shape[1] == 1:

    # bsz X seq_len
    txts = []
    token_lists = []
    pad_idx = src_dict.pad()
    eos_idx = src_dict.eos()

    if tokens.shape[1] == 1:
        # initial inference step, we need to return EOS
        return torch.empty_like(tokens).fill_(ent_eos_idx)
        

    output_entities = torch.empty_like(tokens)

    for i in range(len(tokens)):
        token_list = [src_dict[idx] for idx in tokens[i][1:]]
        token_lists.append(token_list)
        txt = ' '.join(token_list)
        txt = data_utils.process_bpe_symbol(txt, ' ##')
        txts.append(txt)

    docs = list(nlp.pipe(txts))

    for i in range(len(token_lists)):
        doc = docs[i]
        _, alignments = align_tokens(doc, token_lists[i])
        entities = torch.zeros_like(tokens[i])
        entities[0] = ent_eos_idx

        for j in range(1, len(alignments)):
            spacy_token = doc[j]
            ent_type = pad_idx + 1
            if spacy_token.ent_type_ in ENTITY_TYPES:
                ent_type += ENTITY_TYPES[spacy_token.ent_type_]

            for wp_idx in alignments[j]:
                entities[wp_idx] = ent_type
        output_entities[i] = entities

    return output_entities