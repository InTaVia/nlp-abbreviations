from typing import List, Dict
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from abbr_utils import AnnotatedSentence, read_slovene_conll




def _get_multitok_abbrs(tokens_abbr, tokens_exp, labels):
    multitok = defaultdict(list)
    open_exp = []
    curr_abbr = ""
    for i, l in enumerate(labels):
        if len(open_exp) > 0:
            # if l == 'I-ABBR' is the STRICTEST way to catch multi-token abbreviations, however annotations are inconsistent: some are with consecutive [B-ABBR, B-ABBR, ...] instead of [[B-ABBR, I-ABBR, ...]].
            # l != 'O' catches ALL consecutive abbreviations, however this has a lot of FalsePositives and introduces more entropy, is better to treat each abbreviation individually
            if l == 'I-ABBR': 
                open_exp.append(tokens_exp[i])
            elif l == 'O':
                if len(open_exp) > 1:
                    multitok[curr_abbr].append(" ".join(open_exp))
                open_exp = []
        elif l == 'B-ABBR':
            open_exp.append(tokens_exp[i])
            curr_abbr = tokens_abbr[i]
        else:
            continue
    return multitok


def get_dataset_stats(dataset: List[AnnotatedSentence], verbose: bool) -> Dict:
    """Displays the structured data. For debug purposes (put verbose=True), it also returns global statistics information from the whole corpus

    Args:
        dataset (List[AnnotatedSentence]): The list of clean structured corpus examples
        verbose (bool): Decide if print the entire corpus examples or not

    Returns:
        Dict: It includes some relevant corpus stats that might be useful for "global" methods...
    """    
    all_mappings = defaultdict(set)
    mappings_counter = defaultdict(int)
    
    # General Stats Counters
    number_of_sentences, number_of_tokens, number_of_types = 0, 0, 0
    all_tokens = []
    all_abbr_candidates, all_abbr_expansions = [], []
    all_multitok_abbrs = []
    
    for doc in dataset:
        number_of_sentences += 1
        number_of_tokens += len(doc.tokens)
        all_tokens += [tok.text for tok in doc.tokens]
        abbr_sent, abbr_mapping = doc.get_abbreviated()
        for item in abbr_mapping:
            all_abbr_expansions.append(item[1])
            all_abbr_candidates.append(item[2])
            mappings_counter[(item[1],item[2])] += 1
            all_mappings[item[1]].add(item[2])
        if verbose:
            print(doc.get_expanded(original_str=True))
            print(" ".join(abbr_sent))
            print(doc.get_abbr_labels())
            print(doc.get_abbr_labels(tokenized=False))
            print(abbr_mapping)
            print('-------')
    
        abbr_toks, lbls = doc.get_abbr_labels()
        exp_toks = doc.get_expanded(original_str=False)
        multitok = _get_multitok_abbrs(abbr_toks, exp_toks, lbls)
        if len(multitok) > 0:
            all_multitok_abbrs.append(multitok)

    if verbose:
        for k,v in sorted(all_mappings.items(), key= lambda x: len(x[1]), reverse=True):
            print(f"{k} --> {v}")
        print("\n\n")
        for k,v in sorted(mappings_counter.items(), key= lambda x: x[1], reverse=True):
            print(f"{k} --> {v}")
        
    number_of_types = len(set(all_tokens))
    number_abbr_candidates = len(all_abbr_candidates)
    number_unique_abbr_candidates = len(set(all_abbr_candidates))
    number_abbr_expansions = len(all_abbr_expansions)
    number_unique_expansions = len(set(all_abbr_expansions))
    print(f"\tSentences = {number_of_sentences}\n\tTokens = {number_of_tokens}\n\tTypes = {number_of_types}\n\tAbbreviations = {number_abbr_candidates} (unique = {number_unique_abbr_candidates})\n\tExpansions = {number_abbr_expansions} (unique = {number_unique_expansions})")
    print(f"\tAbbr->Expansion Pairs = {len(mappings_counter)}")
    [print(x) for x in all_multitok_abbrs]

    return {
        "exp2abbr_mapping": all_mappings,
        "exp_abbr_pairwise_mapping_counts": mappings_counter
    }

def compare_data_partitions(train_stats: Dict, test_stats: Dict, label: str):
    unique_test_abbrs = set(test_stats['exp_abbr_pairwise_mapping_counts']).difference(train_stats['exp_abbr_pairwise_mapping_counts'])
    unique_candidates = len(set([abbr for _,abbr in unique_test_abbrs]))
    unique_expansions = len(set([exp for exp,_ in unique_test_abbrs]))
    print(f"\nUnique Abbreviation Candidates in {label} (UNSEEN) = {unique_candidates}")
    print(f"Unique Abbreviation Expansions in {label} (UNSEEN) = {unique_expansions}")
    print(f"Unique (Abbreviation -> Expansion) in {label} (UNSEEN) = {len(unique_test_abbrs)}")
    # [print(it) for it in unique_test_abbrs]


def create_token_classification_data(data: List[AnnotatedSentence], use_gold_tokens: bool, output_path: str):
    """This function builds the dataset for STEP 1: Token Classification [ABBR, NO-ABBR] for each Token in the dataset
        We save it in a file so it can be later loaded as a HuggingFace Dataset and make batched experiments with it!

    Args:
        all_data (List[AnnotatedSentence]): The list of clean structured corpus examples
    """
    with open(output_path, "w", encoding='utf-8') as fout:
        for doc in data:
            if use_gold_tokens == False:
                doc.get_abbreviated()
            doc_tokens, doc_labels = doc.get_abbr_labels(tokenized=use_gold_tokens)
            for tok, lbl in zip(doc_tokens, doc_labels):
                fout.write(json.dumps({"token": tok, "gold_label": lbl, "document_id": doc.doc_id}) + "\n")


def create_abbreviation_expansion_data(data: List[AnnotatedSentence], output_path: str, pre_expand_others: bool):
    with open(output_path, "w", encoding='utf-8') as fout:
        for doc in data:
            # We are always taking the CoNLL as Ground Truth here to later compare vs Model predicted abbr candidates!
            # We get back a Dict of tokenized Sentences with
            masked_doc = doc.get_masked_text(sentecized=True, mask_token='<mask>', pre_expand_others=pre_expand_others)
            for sent_id, masked_sent in masked_doc.items():
                data_obj = {'doc_id': doc.doc_id, 'sent_id':sent_id, 'masked_tokens': masked_sent['sentence'], 'candidate': masked_sent['candidate'], 
                            'gold_expansion': masked_sent['gold_expansion'], 'mask_index': masked_sent['mask_index']}
                fout.write(json.dumps(data_obj) + "\n")


# TODO: this one might be better to save as PICKL file and recover completely the object in another script
def save_document_data(data: List[AnnotatedSentence], use_gold_tokens: bool, output_path: str):
    with open(output_path, "w", encoding='utf-8') as fout:
        for doc in data:
            abbr_doc, mapping = doc.get_abbreviated()
            doc_tokens, doc_labels = doc.get_abbr_labels(tokenized=use_gold_tokens)
            data_obj = {
                'document_id': doc.doc_id,
                'original_text': doc.get_expanded(original_str=True),
                'abbreviated_text': abbr_doc,
                'expanded_to_abbreviate': mapping,
                'token_objects': [tok.asdict() for tok in doc.tokens],
                'doc_tokenized': doc_tokens,
                'doc_abbr_token_labels': doc_labels
            }
            fout.write(json.dumps(data_obj) + "\n")


def save_dataset_split_ids(filepath: str, dataset: List[AnnotatedSentence]) -> None:
    doc_dict = defaultdict(list)
    for doc in dataset: 
        doc_dict[doc.doc_id].append(doc.sent_id)
    with open(filepath, "w") as fout:
        json.dump(doc_dict, fout, indent=4)


if __name__ == '__main__':
    #all_data = read_slovene_xml("abbreviations/data/intavia-sl-abbr.xml")
    
    # Returns a List of Sentences because we trust_sentence_boundaries of this version of the file! (sbl-51abbr-expan.conll)
    all_documents = read_slovene_conll("data/sbl-51abbr-expan.conll")

    X_train, X_test = train_test_split(all_documents, test_size=0.2, random_state=4239)
    X_train, X_dev = train_test_split(X_train, test_size=0.125, random_state=4239)
    print(len(X_train), len(X_dev), len(X_test))
    save_dataset_split_ids("data/sbl-51abbr.ids.train", X_train)
    save_dataset_split_ids("data/sbl-51abbr.ids.dev", X_dev)
    save_dataset_split_ids("data/sbl-51abbr.ids.test", X_test)
    
    print(f"\nComputing stats for {len(X_train)} examples in TRAIN")
    train_stats = get_dataset_stats(X_train, verbose=False) # Computes Statistics for the given portion of the dataset
    print(f"\nComputing stats for {len(X_dev)} examples in DEVELOPMENT")
    dev_stats = get_dataset_stats(X_dev, verbose=False)
    print(f"\nComputing stats for {len(X_test)} examples in TEST")
    test_stats = get_dataset_stats(X_test, verbose=False)
    compare_data_partitions(train_stats, test_stats, label="TRAIN vs TEST")
    compare_data_partitions(train_stats, dev_stats, label="TRAIN vs DEV")
    compare_data_partitions(dev_stats, test_stats, label="DEV vs TEST")


    # EXPERIMENT 1: Use a SINGLE TOKEN BINARY CLASSIFIER (BERT) for Abbreviation or No_Abbreviation
    use_gold_tokens=False # This is actually the case for test 'in the wild' for untagged documents. One can always of course automatically pre-tokenize the corpus but maybe this raw option is more robust!
    
    # Train Set for STEP 1
    create_token_classification_data(X_train, use_gold_tokens, output_path='data/sbl-51abbr.tok.train.json')
    save_document_data(X_train, use_gold_tokens, output_path='data/sbl-51abbr.docs.train.json')
    # Development Set for STEP 1
    create_token_classification_data(X_dev, use_gold_tokens, output_path='data/sbl-51abbr.tok.dev.json')
    save_document_data(X_dev, use_gold_tokens, output_path='data/sbl-51abbr.docs.dev.json')
    # Test Set for STEP 1
    create_token_classification_data(X_test, use_gold_tokens, output_path='data/sbl-51abbr.tok.test.json')
    save_document_data(X_test, use_gold_tokens, output_path='data/sbl-51abbr.docs.test.json')

    # EXPERIMENT 2: Save the Train/Test partitions with the documents. This will be used later to [MASK] them and prepare data (from STEP 1) for expansion prediction
    # HERE We Will use the Oracle, assuming we know 100% of the abbreviations. That's why we can create a masked dataset already at this step:
    PRE_EXPAND=True
    if PRE_EXPAND:
        create_abbreviation_expansion_data(X_train, output_path='data/sbl-51abbr.masked.upperbound.preexp.train.json', pre_expand_others=True)
        create_abbreviation_expansion_data(X_dev, output_path='data/sbl-51abbr.masked.upperbound.preexp.dev.json', pre_expand_others=True)
        create_abbreviation_expansion_data(X_test, output_path='data/sbl-51abbr.masked.upperbound.preexp.test.json', pre_expand_others=True)
    else:
        create_abbreviation_expansion_data(X_train, output_path='data/sbl-51abbr.masked.upperbound.train.json', pre_expand_others=False)
        create_abbreviation_expansion_data(X_dev, output_path='data/sbl-51abbr.masked.upperbound.dev.json', pre_expand_others=False)
        create_abbreviation_expansion_data(X_test, output_path='data/sbl-51abbr.masked.upperbound.test.json', pre_expand_others=False)