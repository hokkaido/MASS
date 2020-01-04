import torch
import spacy
from fairseq.data import data_utils
import truecase

from .ner import align_tokens, ENTITY_TYPES

GLOBAL_ITER = 0

nlp = spacy.load('spacy_models/en_core_web_sm_lower', disable=["tagger", "parser"])

        # 0 = PAD
        # 1 = First sentence
        # max_segments = Last sentence
        # max_segments + 1 = EOS

# sequence = size T
def create_segments_for_inference(sequence, split_idx, max_segments):
    eos_idx = max_segments + 1

    if sequence.shape[1] == 1:
        # initial inference step, we need to return EOS
        return torch.empty_like(sequence).fill_(eos_idx)

    segments = torch.zeros_like(sequence)
    
    for segment_token in split_idx:
        segments += sequence.eq(segment_token).type_as(segments)
        
    segments = torch.cumsum(segments, dim=0) + 1
    segments = segments.clamp(0, max_segments)
    segments[:, 0] = eos_idx

    return segments

def create_cheap_ner(tokens, src_dict, ent_pad_idx=0, ent_eos_idx=21):
    if tokens.shape[1] == 1:
        # initial inference step, we need to return EOS
        return torch.empty_like(tokens).fill_(ent_eos_idx)

    output_entities = torch.ones_like(tokens)
    output_entities[:, 0] = ent_eos_idx
    return output_entities

def create_ner_from_output_tokens(tokens, src_dict, ent_pad_idx=0, ent_eos_idx=21):
    global GLOBAL_ITER
    if GLOBAL_ITER % 20 == 0:
        global nlp
        nlp = spacy.load('spacy_models/en_core_web_sm_lower', disable=["tagger", "parser"])

    # bsz X seq_len
    txts = []
    token_lists = []

    if tokens.shape[1] == 1:
        # initial inference step, we need to return EOS
        return torch.empty_like(tokens).fill_(ent_eos_idx)
        

    output_entities = torch.empty_like(tokens)

    for i in range(len(tokens)):
        token_list = [src_dict[idx] for idx in tokens[i][1:]]
        token_lists.append(token_list)
        txt = ' '.join(token_list)
        txt = data_utils.process_bpe_symbol(txt, ' ##')
        txt = truecase.get_true_case(txt)
        txts.append(txt)

    docs = list(nlp.pipe(txts))

    for i in range(len(token_lists)):
        doc = docs[i]
        _, alignments = align_tokens(doc, token_lists[i])
        entities = torch.zeros_like(tokens[i])
        entities[0] = ent_eos_idx

        for j in range(1, len(alignments)):
            spacy_token = doc[j]
            ent_type = 1 #NONE
            if spacy_token.ent_type_ in ENTITY_TYPES:
                ent_type += ENTITY_TYPES[spacy_token.ent_type_]

            for wp_idx in alignments[j]:
                entities[wp_idx] = ent_type
        output_entities[i] = entities

    GLOBAL_ITER += 1
    return output_entities

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)