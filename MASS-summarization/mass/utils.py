import torch
from fairseq.data import data_utils

from .segmented_dataset import create_ner
from .ner import align_tokens, ENTITY_TYPES

import spacy

spacy.require_gpu()

nlp = spacy.load("en_core_web_sm")

def create_ner_from_output_tokens(tokens, src_dict):

    #if tokens.shape[1] == 1:

    # bsz X seq_len
    txts = []
    token_lists = []
    output_entities = torch.zeros_like(tokens)
    pad_idx = src_dict.pad()

    for i in range(len(tokens)):
        token_list = [src_dict[idx] for idx in tokens[i]]
        token_lists.append(token_list)
        txt = ' '.join(token_list)
        print(txt)
        txt = data_utils.process_bpe_symbol(txt, ' ##')
        txts.append(txt)

    docs = list(nlp.pipe(txts))

    for i in range(len(tokens)):
        doc = docs[i]
        _, alignments = align_tokens(doc, token_list[i])
        print(alignments)
        entities = torch.zeros_like(tokens[i])
        for j in range(len(alignments)):
            spacy_token = doc[j]
            ent_type = pad_idx + 1
            if spacy_token.ent_type_ in ENTITY_TYPES:
                ent_type += ENTITY_TYPES[spacy_token.ent_type_]

            for wp_idx in alignments[j]:
                entities[wp_idx] = ent_type
        output_entities[i] = entities
    return output_entities