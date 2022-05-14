import copy, re
from dataclasses import dataclass
from lxml import etree
from collections import defaultdict, deque
import logging
from enum import Enum
from typing import List, Tuple, NamedTuple
from transformers import pipeline
from transformers import AutoModelForMaskedLM

import Levenshtein as lev


class ExpansionInstance(NamedTuple):
    doc_id: str
    sent_id: str 
    query: str
    gold: str
    prediction: str
    pred_candidates: List 
    abbrev: str
    mask_index: int

class CandidateLM(NamedTuple):
    text: str
    gold_abbr: str
    score_lm: float
    levenshtein: float
    jaccard_n1: float
    jaccard_n2: float

class BestFitPolicy(Enum):
    HIGHEST_PROB = 1
    FIRST_LETTER = 2
    LEVENSHTEIN = 3
    JACCARD_N1 = 4
    JACCARD_N2 = 5


class ConllToken:
    def __init__(self, row: str, sent_id: str = "[UNK]", tok_id_offset: int = 0, splitter='\t') -> None:
        self.splitter = splitter
        self.raw = row
        fields = row.split(splitter)
        self.sent_id = sent_id # This is valid for gold/silver data. Set to UNK otherwise
        self.index = int(fields[0]) - 1 + tok_id_offset # The offset is to adapt it to a Global Document Indexing. Still NOT Implemented properly
        self.text = fields[1]
        self.lemma = fields[2]
        self.pos = fields[3]
        self.case = fields[4]
        self.morph = fields[5]
        self.ner, self.abbr, self.space_after = self._decompose_bio_tag(fields[9])
        self.starts_with_uppercase = self.text[0].isupper()
    
    def asconll(self):
        return self.raw

    def _decompose_bio_tag(self, complex_tag: str) -> str:
        # ABBR=O|NER=O|SpaceAfter=No
        x = complex_tag.split('|')
        abbr = x[0].split('=')[1].strip()
        ner = x[1].split('=')[1].strip()
        if len(x) > 2:
            space_after = x[2].split('=')[1]
        else:
            space_after = 'Yes'
        return ner, abbr, space_after

    def asdict(self):
        return {'sent_id': self.sent_id, 'index': self.index, 'text': self.text, 'lemma': self.lemma, 'pos': self.pos,
                'case': self.case, 'morph': self.morph, 'ner': self.ner, 'abbr': self.abbr, 'space_after': self.space_after}


@dataclass
class AnnotatedSentence:
    doc_id: str
    sent_id: str
    text: str
    tokens: List[ConllToken] = None
    abbr_mapping = None
    model_predictions = None

    def get_expanded(self, original_str=False):
        if original_str:
            return self.text
        else:
            return [tok.text for tok in self.tokens]
    
    def get_abbreviated(self) -> Tuple[List, List]:
        abbr_sent, mapping = [], []
        for tok in self.tokens:
            if tok.abbr == 'O':
                abbr_sent.append(tok.text)
            elif tok.abbr.startswith('B-'):
                my_abbr = tok.abbr[2:]
                abbr_sent.append(my_abbr)
                mapping.append((tok.index, tok.text, my_abbr))
            elif tok.abbr.startswith('I-'):
                abbr_sent.append("[CONT]")
        assert len(abbr_sent) == len(self.tokens)
        self.abbr_mapping = mapping
        return abbr_sent, mapping

    def get_abbr_labels(self, tokenized: bool = True) -> List:
        tokens, labels = [], []
        if tokenized:
            for tok in self.tokens:
                if tok.abbr == 'O':
                    labels.append('O')
                    tokens.append(tok.text)
                elif tok.abbr.startswith('B-'):
                    labels.append('B-ABBR')
                    tokens.append(tok.abbr[2:])
                elif tok.abbr.startswith('I-'):
                    labels.append('I-ABBR') # This is to "skip" multiword-abbreviations (anno domini -> an.), that's why we only take the B- token. There are 16 such cases in ALL data (before split).
                    tokens.append(tok.abbr[2:])
            assert len(labels) == len(self.tokens)
        elif self.abbr_mapping is not None:
            naive_tokens = self.text.split()
            naive_abbr_tokens = []
            if len(self.abbr_mapping) == 0: return naive_tokens, ['O' for t in naive_tokens]
            keywords = deque([exp for ix,exp,abbr in self.abbr_mapping]) # pop function == O(1)!
            abbrs = [abbr for ix,exp,abbr in self.abbr_mapping]
            if len(keywords) == 0: return naive_tokens, ['O' for t in naive_tokens]
            substitute_ix = 0
            for tok in naive_tokens:
                clean_tok = re.sub(r'[^\w\s]',"",tok)
                try:
                    if clean_tok == keywords[0]:
                        labels.append('ABBR')
                        keywords.popleft()
                        naive_abbr_tokens.append(abbrs[substitute_ix])
                        substitute_ix += 1
                    else:
                        labels.append('O')
                        naive_abbr_tokens.append(tok)
                except:
                    labels.append('O') # There are no more keywords so the rest is 'O' by default...
                    naive_abbr_tokens.append(tok)
                
            assert len(labels) == len(naive_tokens)
            tokens = naive_abbr_tokens
        else:
            raise Exception("Please put tokenized=True, or run get_abbr_document() first, to have the mapping needed for this function!")
        return tokens, labels
    
    def get_masked_text(self, sentecized=True, mask_token='[MASK]', pre_expand_others=False):
        masked_sentences = defaultdict(list)
        masked_correspondence = defaultdict(list) # Abbreviations that where masked IN ORDER i.e. masked_correspondence[n] is the nth [MASK] token in the sentence
        if sentecized == True:
            for token in self.tokens:
                if token.abbr == 'O':
                    masked_sentences[token.sent_id].append(token.text)
                else:
                    masked_sentences[token.sent_id].append(mask_token)
                    masked_correspondence[token.sent_id].append((token.index, token.abbr[2:], token.text)) # 2: Is to strip the B- or I- part of the tag!
        else:
            full_doc = []
            for token in self.tokens:
                if token.abbr == 'O':
                    full_doc.append(token.text)
                else:
                    full_doc.append(mask_token)
                    masked_correspondence[self.doc_id].append((token.index, token.abbr[2:], token.text))
            masked_sentences[self.doc_id] = full_doc
        # Expand to 1-Example-Per-Mask (note that this fileters away the sentences that had ZERO abbreviations)
        expanded_masked_sentences = dict()
        for id_key, mask_list in masked_correspondence.items():
            to_be_expanded = masked_sentences[id_key]
            for mask_ix in range(len(mask_list)):
                freeze_ix, freeze_abbr, freeze_expan = mask_list[mask_ix]
                substitutes = [mask_list[x] for x in range(len(mask_list)) if x != mask_ix]
                new = copy.deepcopy(to_be_expanded)
                for sub_ix, sub_abbr, sub_exp in substitutes:
                    if pre_expand_others:
                        new[sub_ix] = sub_exp
                    else:
                        new[sub_ix] = sub_abbr
                assert new.count(mask_token) == 1
                expanded_masked_sentences[f"{id_key}.{mask_ix}"] = {'sentence': new, 'candidate': freeze_abbr, 'gold_expansion': freeze_expan, 'mask_index': freeze_ix}

        return expanded_masked_sentences
    
    def get_as_conll(self): #WARN: This method has NOT BEEN TESTED!!!
        final_conll = []
        prev_sent = self.tokens[0].sent_id
        for tok in self.tokens:
            if tok.sent_id == prev_sent:
                final_conll.append(tok.asconll())
            else:
                prev_sent = tok.sent_id
                final_conll.append(["\n"])
                final_conll.append(tok.asconll())
        return "\n".join(final_conll)



def choose_best_fit(possible_fits: List[CandidateLM], policy: BestFitPolicy, topk: int) -> List[str]:
    assert topk > 0
    if topk > len(possible_fits): topk = len(possible_fits)
    top_fits = []
    if policy == BestFitPolicy.FIRST_LETTER:
        for p in possible_fits:
            if len(p.text) > 0 and p.text[0].lower() == p.gold_abbr[0].lower():
                top_fits.append(p.text)
        if len(top_fits) == 0: return ["<NO-FIT>"]
        return top_fits[:topk]
    elif policy == BestFitPolicy.HIGHEST_PROB:
        top_fits = [p.text for p in possible_fits[:topk]]
        return top_fits
    elif policy == BestFitPolicy.LEVENSHTEIN:
        top_fits = [p.text for p in sorted(possible_fits, key=lambda x: x.levenshtein)[:topk]]
        return top_fits
    elif policy == BestFitPolicy.JACCARD_N1:
        top_fits = [p.text for p in sorted(possible_fits, key=lambda x: x.jaccard_n1)[:topk]]
        return top_fits
    elif policy == BestFitPolicy.JACCARD_N2:
        # top_fits = [p.text for p in sorted(possible_fits, key=lambda x: x.jaccard_n2)[:topk]]
        top_fits = []
        for p in possible_fits:
            if len(p.text) > 0 and p.text[0].lower() == p.gold_abbr[0].lower():
                top_fits.append(p)
        if len(top_fits) == 0: return ["<NO-FIT>"]
        return [p.text for p in sorted(top_fits, key=lambda x: x.jaccard_n2)[:topk]]
    else:
        raise NotImplementedError



def jaccard_distance(str1, str2, ngram=1):
    if ngram == 1:
        A, B = set(str1), set(str2)
    else:
        A = set([str1[i:i+ngram] for i in range(len(str1)+1-ngram)])
        B = set([str2[i:i+ngram] for i in range(len(str2)+1-ngram)])
    union_len = len(A.union(B))
    jacc = len(A.intersection(B)) / len(A.union(B)) if union_len > 0 else 0
    return float(1 - jacc)





def read_slovene_xml(filepath: str) -> List:
    annotations = []
    tree = etree.parse(filepath).getroot()
    for elem in tree:
        if elem.get("type") == "entry":
            abbreviations, expansions = [], []
            text, expanded_text = [], []

            text_id = elem.get('{http://www.w3.org/XML/1998/namespace}id')
            pelem = elem.find('{http://www.tei-c.org/ns/1.0}p')
            for p in pelem.iter('{http://www.tei-c.org/ns/1.0}choice'):
                try:
                    abb = p.find('{http://www.tei-c.org/ns/1.0}abbr').text
                    exp = p.find('{http://www.tei-c.org/ns/1.0}expan').text
                    abbreviations.append(abb)
                    expansions.append(exp)
                except:
                    pass
            
            for p in pelem.itertext():
                if p in abbreviations:
                    text.append(p)
                elif p in expansions:
                    expanded_text.append(p)
                else:
                    text.append(p)
                    expanded_text.append(p)

            annotations.append((text_id, ''.join(text), ''.join(expanded_text), abbreviations, expansions, []))
    return annotations



def read_slovene_conll(filepath: str) -> List[AnnotatedSentence]:
    """[summary]

    Args:
        filepath (str): Filepath to the CoNLL File
        trust_sentence_boundaries (bool): If True then each Document == A single sentence, 
                                            otherwise each document contains all tokens with the same Doc_ID

    Returns:
        List[AnnotatedSentence]: List of Annotated Documents
    """    
    sentences = []
    buffer = []
    with open(filepath) as f:
        for line in f.readlines():
            if line.startswith('# sent_id = '):
                sent_id = line[12:].rstrip('\n')
            elif line.startswith('# newdoc id ='):
                doc_id = line[14:].rstrip('\n')
            elif line.startswith('# text = '):
                text = line[9:].rstrip('\n')
            elif line.startswith('# '):
                pass
            elif len(line) > 1:
                buffer.append(ConllToken(line, sent_id))
            else:
                sentences.append(AnnotatedSentence(doc_id.lstrip(), sent_id.lstrip(), text.lstrip(), tokens=buffer))
                buffer = []
    return sentences


def get_dataset_partition(dataset: List[AnnotatedSentence], partition_dict: str) -> List[AnnotatedSentence]:
    partition_docs: List[AnnotatedSentence] = []
    for doc in dataset:
        if doc.doc_id in partition_dict:
            if doc.sent_id in partition_dict[doc.doc_id]:
                partition_docs.append(doc)
    return partition_docs


def get_lang_model_predictions(dataset: List, model_name: str, tokenizer_name: str, top_k: int = 5, gpu_ix: int = -1) -> List:
    prediction_list = []
    prediction_dict = defaultdict(list)
    total_examples = len(dataset)
    skept_examples = []
    if gpu_ix >= 0:
        pipe = pipeline('fill-mask', model=model_name, tokenizer=model_name, device=gpu_ix)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        pipe = pipeline('fill-mask', model=model, tokenizer=tokenizer_name)
    for k, row in enumerate(dataset):
        masked_sent = " ".join(row['masked_tokens'])
        logging.info(f"\n------- {k+1}/{total_examples} ---------\nLooking for: {row['candidate']} --> {row['gold_expansion']}\n{masked_sent}\n\n")
        best_fit = "<NO-FIT>"
        try:
            model_predictions = pipe(masked_sent, top_k=top_k) # ADDING PARAM targets= ['milk'] RETURNS [{'sequence': 'I ate bacon and milk for breakfast', 'score': 0.010439875535666943, 'token': 6831, 'token_str': 'milk'}]
        except:
            skept_examples.append(row['sent_id'])
            continue
        possible_fits = []
        for res in model_predictions:
            logging.info(res['sequence'])
            logging.info(f"TOK: {res['token_str']} | SCORE: {res['score']}")
            # For now save all predictions, and later we choose a best_fit based on some kind of policy...
            possible_fits.append(CandidateLM(res['token_str'], row['candidate'], res['score'], 
                                            lev.distance(row['candidate'], res['token_str']), 
                                            jaccard_distance(row['candidate'], res['token_str'], ngram=1),
                                            jaccard_distance(row['candidate'], res['token_str'], ngram=2)
                                            ))
        better_fits = choose_best_fit(possible_fits, policy=BestFitPolicy.FIRST_LETTER, topk=5)
        better_fits_jaccard = choose_best_fit(possible_fits, policy=BestFitPolicy.JACCARD_N2, topk=10)
        best_fit = better_fits[0]
        # prediction_dict[row['candidate']].append(best_fit)
        prediction_dict[row['candidate']] += better_fits_jaccard
        prediction_list.append(ExpansionInstance(row['doc_id'], row['sent_id'], masked_sent, row['gold_expansion'], best_fit, possible_fits, row['candidate'], row['mask_index']))
    print(skept_examples)
    return prediction_list, prediction_dict
