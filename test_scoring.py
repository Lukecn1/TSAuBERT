import numpy as np
import math
import unittest
from nltk import sent_tokenize
from scipy.spatial.distance import cosine

from score import get_tsaubert, get_tsaubert_scores
from utils import get_bert_model
from encode import pool_vectors, combine_word_piece_vectors, get_ngram_embedding_vectors, get_embedding_vectors, get_embeddings


class testScoringFunctions(unittest.TestCase):

    def test_get_tsaubert(self):

        candidate_vectors = [[3.4, 6.2, 4.5, -5.5], 
                             [4.3, 2.6, 5.4, 5.5], 
                             [2.4, 3.2, 6.5, 8.5]]
        
        reference_vectors = [[2.3, 7.8, 1.5, -1.8], 
                             [9.9, -1.6, 2.4, 5.5], 
                             [6.4, 3.2, 7.8, 5.1], 
                             [1.4, -7.3, 1.5, -1.5]]

        cand_ort_vectors = [[1, 0, 1, 0]]
        ref_ort_vectors  = [[0, 1, 0, 1]]
        cand_eq_vec = [[0, 1, 0, 1]]   

        
        results_eq = get_tsaubert(cand_eq_vec, ref_ort_vectors, 'argmax')
        results_ort = get_tsaubert(cand_ort_vectors, ref_ort_vectors, 'argmax')
        results_mean_function = get_tsaubert(candidate_vectors, reference_vectors, 'mean')
        results_argmax_function = get_tsaubert(candidate_vectors, reference_vectors, 'argmax')
        
        # Calculations of the scores with the candidate and reference vectors manually
        results_argmax_manual = (0.9101563472457149, 0.6034360551295096, 0.7257187005843395) 
        results_mean_manual =  (0.3661780061962483, 0.36617800619624835, 0.36617800619624835)
        
        self.assertAlmostEqual(results_mean_function[0], results_mean_manual[0])
        self.assertAlmostEqual(results_mean_function[1], results_mean_manual[1])
        self.assertAlmostEqual(results_mean_function[2], results_mean_manual[2])
        self.assertAlmostEqual(results_argmax_function[0], results_argmax_manual[0])
        self.assertAlmostEqual(results_argmax_function[1], results_argmax_manual[1])
        self.assertAlmostEqual(results_argmax_function[2], results_argmax_manual[2])
        self.assertAlmostEqual(results_eq[0], 1.0)
        self.assertAlmostEqual(results_eq[1], 1.0)
        self.assertAlmostEqual(results_eq[2], 1.0)
        self.assertEqual(results_ort[0], 0.0)
        self.assertEqual(results_ort[1], 0.0)


    def test_score_consistency(self):
        cand_1 = ['This is a test for whether the same candidate summary gets the same score. With a second sentence.']
        cand_2 = ['This is a test for whether the same candidate summary gets the same score. With a second sentence.']
        ref = ['Here is the reference summary for testing scoring consisitency.']

        self.assertEqual(cand_1[0], cand_2[0])

        p_1, r_1, f1_1 = get_tsaubert_scores(cand_1, 
                                   ref, 
                                   scoring_approach = 'mean', 
                                   model_name = 'bert-base-uncased', 
                                   layer= 11, 
                                   n_gram_encoding= 2,
                                   pool_word_pieces= True, 
                                   language= 'english')

        p_2, r_2, f1_2 = get_tsaubert_scores(cand_2, 
                                         ref, 
                                         scoring_approach = 'mean', 
                                         model_name = 'bert-base-uncased', 
                                         layer= 11, 
                                         n_gram_encoding= 2,
                                         pool_word_pieces= True, 
                                         language= 'english')
        

        p_1_sent, r_1_sent, f1_1_sent = get_tsaubert_scores(cand_1, 
                                                        ref, 
                                                        scoring_approach = 'argmax', 
                                                        model_name = 'deepset/sentence_bert', 
                                                        layer= 11,                                                 
                                                        n_gram_encoding=  None,
                                                        pool_word_pieces= False, 
                                                        language= 'english')

        p_2_sent, r_2_sent, f1_2_sent = get_tsaubert_scores(cand_2, 
                                                        ref, 
                                                        scoring_approach = 'argmax', 
                                                        model_name = 'deepset/sentence_bert', 
                                                        layer= 11, 
                                                        n_gram_encoding= None,
                                                        pool_word_pieces= False, 
                                                        language= 'english')

        self.assertEqual(p_1[0], p_2[0])
        self.assertEqual(r_1[0], r_2[0])
        self.assertEqual(f1_1[0], f1_2[0])

        self.assertEqual(p_1_sent[0], p_2_sent[0])
        self.assertEqual(r_1_sent[0], r_2_sent[0])
        self.assertEqual(f1_1_sent[0], f1_2_sent[0])
        



    def test_control_flow(self):

        candidate_summaries = ['First candidate summary for testing. Another sentence for testing purposes. The final phrase is written here.', 
                               'Second candidate summary is written here. It only consists of two sentences.', 
                                'The third and final candidate summary is here. It has more than two sentences. Hence the third text sequence.']
        
        reference_summaries = [ 'Here is the first sentence of the reference summary. Only two individual sentences for this summary.', 
                                'Start of the second reference. Testing the controlflow of the embedding functions.',
                                'Lastly a single sentence reference summary.']


        precision_scores_mean, recall_scores_mean, f1_scores_mean = get_tsaubert_scores(candidate_summaries, 
                                                                     reference_summaries, 
                                                                     scoring_approach = 'mean', 
                                                                     model_name = 'bert-base-uncased', 
                                                                     layer= 11, 
                                                                     n_gram_encoding= 2,
                                                                     pool_word_pieces= True, 
                                                                     language= 'english')
    

        self.assertEqual(len(precision_scores_mean), 3)
        self.assertEqual(len(recall_scores_mean), 3)
        self.assertEqual(len(f1_scores_mean), 3)
        self.assertNotEqual(f1_scores_mean[0], f1_scores_mean[1])
        self.assertNotEqual(f1_scores_mean[1], f1_scores_mean[2])
        self.assertNotEqual(f1_scores_mean[0], f1_scores_mean[2])

        
        precision_scores_max, recall_scores_max, f1_scores_max = get_tsaubert_scores(candidate_summaries, 
                                                                     reference_summaries, 
                                                                     scoring_approach = 'argmax', 
                                                                     model_name = 'bert-base-uncased', 
                                                                     layer= 11, 
                                                                     n_gram_encoding= 2,
                                                                     pool_word_pieces= True, 
                                                                     language= 'english')
                                 

        self.assertNotEqual(f1_scores_max[0], f1_scores_max[1])
        self.assertNotEqual(f1_scores_max[1], f1_scores_max[2])
        self.assertNotEqual(f1_scores_max[0], f1_scores_max[2])
        self.assertNotEqual(precision_scores_mean[0], precision_scores_max[0])
        self.assertNotEqual(recall_scores_mean[0], recall_scores_max[0])
        self.assertNotEqual(f1_scores_mean[0], f1_scores_max[0])