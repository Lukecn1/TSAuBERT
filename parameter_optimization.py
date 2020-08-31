import hyperopt as hpt
from hyperopt import fmin, hp
import pandas as pd
import numpy as np
import score
import encode 
import utils
from nltk import sent_tokenize

df = pd.read_csv('C:/Users/Lukas/ITU/Master_Thesis/Metric_paper/Data/TAC/TAC_preprocessed.csv', sep= '\t')
df = df.sample(frac = 0.01, random_state = 11).reset_index()

cand_sums = df['Summary']
ref1 = df['refsum1']
ref2 = df['refsum2']
ref3 = df['refsum3']
ref4 = df['refsum4']

model, tokenizer = utils.get_bert_model('bert-base-uncased')

# Note that we are only searching in n-gram options, as this vastly more computationally expensive compared to sentence-level
# and a relatively inexpensive grid-search can be performed for sentence level optimization. 
search_space = {'layer': hp.choice('layer', [7,8,9,10,11,12]), 
                'n_gram_size': hp.choice('n_gram_size', [2,3,4,5,6,7,8]),
                'scoring_approach': hp.choice('scoring_approach', ['mean', 'argmax']),                
                'wordpiece_pooling': hp.choice('wordpiece_pooling', ['True', 'False'])
                }


def get_scores(args):
    """
    Returns a list of the TSAuBERT scores for the given parameters in the args dictionary 
    """
    layer = args['layer']
    n_gram = args['n_gram_size']
    scoring = args['scoring_approach']
    wp = args['wordpiece_pooling']

    final_precision_scores = []
    final_recall_scores = []
    final_f1_scores = []

    for i, _ in enumerate(cand_sums, 0):
        cand_sum = sent_tokenize(cand_sums[i], language= 'english')
        ref1_sum = sent_tokenize(ref1[i], language= 'english')
        ref2_sum = sent_tokenize(ref2[i], language= 'english')
        ref3_sum = sent_tokenize(ref3[i], language= 'english')
        ref4_sum = sent_tokenize(ref4[i], language= 'english')

        cand_vecs, cand_tokens = encode.get_embeddings(cand_sum, model, layer, tokenizer)
        ref1_vecs, ref1_tokens = encode.get_embeddings(ref1_sum, model, layer, tokenizer) 
        ref2_vecs, ref2_tokens = encode.get_embeddings(ref2_sum, model, layer, tokenizer)
        ref3_vecs, ref3_tokens = encode.get_embeddings(ref3_sum, model, layer, tokenizer)
        ref4_vecs, ref4_tokens = encode.get_embeddings(ref4_sum, model, layer, tokenizer)

        final_cand_vecs = encode.get_ngram_embedding_vectors(cand_vecs, n_gram, wp, cand_tokens)
        final_ref1_vecs = encode.get_ngram_embedding_vectors(ref1_vecs, n_gram, wp, ref1_tokens)
        final_ref2_vecs = encode.get_ngram_embedding_vectors(ref2_vecs, n_gram, wp, ref2_tokens)
        final_ref3_vecs = encode.get_ngram_embedding_vectors(ref3_vecs, n_gram, wp, ref3_tokens)
        final_ref4_vecs = encode.get_ngram_embedding_vectors(ref4_vecs, n_gram, wp, ref4_tokens)

        p_1, r_1, f1_1 = score.get_tsaubert(final_cand_vecs, final_ref1_vecs, scoring)
        p_2, r_2, f1_2 = score.get_tsaubert(final_cand_vecs, final_ref2_vecs, scoring)
        p_3, r_3, f1_3 = score.get_tsaubert(final_cand_vecs, final_ref3_vecs, scoring)
        p_4, r_4, f1_4 = score.get_tsaubert(final_cand_vecs, final_ref4_vecs, scoring)

        final_precision_scores.append((p_1 + p_2 + p_3 + p_4) / 4)
        final_recall_scores.append((r_1 + r_2 + r_3 + r_4) / 4) 
        final_f1_scores.append((f1_1 + f1_2 + f1_3 + f1_4) / 4) 

    return final_f1_scores

print(len(df))


def objective_function(args):
    """
    Returns the mean correlation (using Pearsons correlation) wrt. the human judgements.
    The score for each candidate summary is the mean score f1-score for the 4 reference summaries. 
    (Score is multiplied as negative, given that the 'fmin' function in hyperopt selects minimal)
    """
    print(args)
    summaries_scores = pd.Series(get_scores(args))

    pyramid_correlation = df['modified pyramid score'].corr(summaries_scores, method = 'pearson')
    readability_correlaton = df['linguistic quality'].corr(summaries_scores, method = 'pearson')
    responsiveness_correlation = df['overall responsiveness'].corr(summaries_scores, method = 'pearson')
    print(pyramid_correlation, readability_correlaton, responsiveness_correlation)

    score = (pyramid_correlation + readability_correlaton + responsiveness_correlation) / 3

    print(score)

    return score * -1

best = fmin(objective_function,
            space= search_space,
            algo = hpt.tpe.rand.suggest,
            max_evals = 10)

print('Best result parameters:')
print(hpt.space_eval(search_space, best))
"""
"""