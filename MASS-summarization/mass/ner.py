import numpy as np

"""
PERSON	People, including fictional.
NORP	Nationalities or religious or political groups.
FAC	Buildings, airports, highways, bridges, etc.
ORG	Companies, agencies, institutions, etc.
GPE	Countries, cities, states.
LOC	Non-GPE locations, mountain ranges, bodies of water.
PRODUCT	Objects, vehicles, foods, etc. (Not services.)
EVENT	Named hurricanes, battles, wars, sports events, etc.
WORK_OF_ART	Titles of books, songs, etc.
LAW	Named documents made into laws.
LANGUAGE	Any named language.
DATE	Absolute or relative dates or periods.
TIME	Times smaller than a day.
PERCENT	Percentage, including ”%“.
MONEY	Monetary values, including unit.
QUANTITY	Measurements, as of weight or distance.
ORDINAL	“first”, “second”, etc.
CARDINAL	Numerals that do not fall under another type.
"""

ENTITY_TYPES = {
    'PERSON': 1,
    'NORP': 2,
    'FAC': 3,
    'ORG': 4,
    'GPE': 5,
    'LOC': 6,
    'PRODUCT': 7,
    'EVENT': 8,
    'WORK_OF_ART': 9,
    'LAW': 10,
    'LANGUAGE': 11,
    'DATE': 12,
    'TIME': 13,
    'PERCENT': 14,
    'MONEY	': 15,
    'QUANTITY': 16,
    'ORDINAL': 17,
    'CARDINAL': 18,
    'RUBBISH': 19,
}

def clean_wp_token(token):
    return token.replace("##", "", 1).strip()

def flatten_list(nested):
    """Flatten a nested list."""
    flat = []
    for x in nested:
        flat.extend(x)
    return flat

def align_tokens(doc, wp_tokens, *, offset=0):
        spacy_tokens = [w.text.rstrip().lower() for w in doc]
        new_wp_tokens = [clean_wp_token(t) for t in wp_tokens]
        assert len(wp_tokens) == len(new_wp_tokens)
        align = align_word_pieces(spacy_tokens, new_wp_tokens, retry=False)
        for indices in align:
            for i in range(len(indices)):
                indices[i] += offset
        return wp_tokens, align

def align_word_pieces(spacy_tokens, wp_tokens, retry=True):
    """Align tokens against word-piece tokens. The alignment is returned as a
    list of lists. If alignment[3] == [4, 5, 6], that means that spacy_tokens[3]
    aligns against 3 tokens: wp_tokens[4], wp_tokens[5] and wp_tokens[6].
    All spaCy tokens must align against at least one element of wp_tokens.
    """
    spacy_tokens = list(spacy_tokens)
    wp_tokens = list(wp_tokens)
    if not wp_tokens:
        return [[] for _ in spacy_tokens]
    elif not spacy_tokens:
        return []
    # Check alignment
    spacy_string = "".join(spacy_tokens).lower()
    wp_string = "".join(wp_tokens).lower()
    if not spacy_string and not wp_string:
        return None
    if spacy_string != wp_string:
        if retry:
            # Flag to control whether to apply a fallback strategy when we
            # don't align, of making more aggressive replacements. It's not
            # clear whether this will lead to better or worse results than the
            # ultimate fallback strategy, of calling the sub-tokenizer on the
            # spaCy tokens. Probably trying harder to get alignment is good:
            # the ultimate fallback actually *changes what wordpieces we
            # return*, so we get (potentially) different results out of the
            # transformer. The more aggressive alignment can only change how we
            # map those transformer features to tokens.
            spacy_tokens = [alpha_re.sub("", t) for t in spacy_tokens]
            wp_tokens = [alpha_re.sub("", t) for t in wp_tokens]
            spacy_string = "".join(spacy_tokens).lower()
            wp_string = "".join(wp_tokens).lower()
            if spacy_string == wp_string:
                return _align(spacy_tokens, wp_tokens)
        # If either we're not trying the fallback alignment, or the fallback
        # fails, we return None. This tells the wordpiecer to align by
        # calling the sub-tokenizer on the spaCy tokens.
        return None
    output = _align(spacy_tokens, wp_tokens)
    if len(set(flatten_list(output))) != len(wp_tokens):
        return None
    return output


def _align(seq1, seq2):
    # Map character positions to tokens
    map1 = _get_char_map(seq1)
    map2 = _get_char_map(seq2)
    # For each token in seq1, get the set of tokens in seq2
    # that share at least one character with that token.
    alignment = [set() for _ in seq1]
    unaligned = set(range(len(seq2)))
    for char_position in range(map1.shape[0]):
        i = map1[char_position]
        j = map2[char_position]
        alignment[i].add(j)
        if j in unaligned:
            unaligned.remove(j)
    # Sort, make list
    output = [sorted(list(s)) for s in alignment]
    # Expand alignment to adjacent unaligned tokens of seq2
    for indices in output:
        if indices:
            while indices[0] >= 1 and indices[0] - 1 in unaligned:
                indices.insert(0, indices[0] - 1)
            last = len(seq2) - 1
            while indices[-1] < last and indices[-1] + 1 in unaligned:
                indices.append(indices[-1] + 1)
    return output

def _get_char_map(seq):
    char_map = np.zeros((sum(len(token) for token in seq),), dtype="i")
    offset = 0
    for i, token in enumerate(seq):
        for j in range(len(token)):
            char_map[offset + j] = i
        offset += len(token)
    return char_map