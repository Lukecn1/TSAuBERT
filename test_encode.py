import numpy as np
import unittest
from nltk import sent_tokenize
from scipy.spatial.distance import cosine
from utils import get_bert_model
from encode import pool_vectors, combine_word_piece_vectors, get_ngram_embedding_vectors, get_embedding_vectors, get_embeddings

"""
Unit tests for the encode functions. 

Each function have at least 1 dedicated test, most have a couple. 

The tests are designed to ensure that the following items are done consistently correctly:

    1) Embedding vectors are produced in accordance with the provided sentences
        1.1) Correct number of vectors for both sentence and word level encodings -  DONE
        1.2) That indicies in each function line up with each other - DONE

    2) Function that pool together vectors take the correct vectors as input.
        2.1) Pooling is done correctly, given the specific pooling strategy - DONE
        2.2) Vectors for [CLS], [SEP] and '.' are not utilized in the comparisons. - DONE 
        2.3) That relevant word vectors are not omitted or lost in the process - DONE
        2.4) That vectors for paddings are never passed to the scoring functions - DONE

    3) That the flow of summaries throughout yields the correct results
        3.1) The chain of function call holds for the correct parameters - DONE
        3.2) That the original order of the input summaries is kept post encoding  - DONE

    4) That the above functionality is kept when changing the BERT model
        4.1) mBERT and other languages than english
        4.2) Danish BERT 
        4.3) sentence-bert
"""

class testEncodeFunctions(unittest.TestCase):

    def test_pool_vectors(self):
        vectors_zero = [[0, 0, -0, -0], [0, 0,-0, -0], [0, 0, -0, -0], [0, 0, -0, -0]]
        vectors_float = [[3.4, 6.2, 4.5, -5.5], [4.3, 2.6, 5.4, 5.5], [2.4, 3.2, 6.5, 8.5], [5.4, -7.2, 2.5, -1.5]]

        self.assertEqual(len(pool_vectors(vectors_float)), 4) # That the dimensionality of the input vectors are kept
        self.assertEqual(sum(pool_vectors(vectors_zero)), 0)  # Vectors with zero values
        self.assertAlmostEqual(pool_vectors(vectors_float)[0], 3.875)
        self.assertAlmostEqual(pool_vectors(vectors_float)[1], 1.2)
        self.assertAlmostEqual(pool_vectors(vectors_float)[2], 4.725)
        self.assertAlmostEqual(pool_vectors(vectors_float)[3], 1.75)


    def test_combine_word_piece_vectors(self):
        vectors_no_split = [[3.4, 6.2, 4.5, -5.5], [3.3, 6.2, 4.5, -5.5], [4.3, 2.6, 5.4, 5.5], [5.4, 3.2, 6.5, 8.5], [5.4, -7.2, 2.5, -1.5]]
        tokens_no_split  = ['[CLS]', 'Token', 'ize', 'word', 'piece', '.', '[SEP]']

        vectors_split_2 = [[3.4, 6.2, 4.5, -5.5], [3.3, 6.2, 4.5, -5.5], [4.3, 2.6, 5.4, 5.5], [5.4, 3.2, 6.5, 8.5], [5.4, -7.2, 2.5, -1.5], [3.4, 6.2, 4.5, -5.5], [3.4, 6.2, 4.5, -5.5]]
        tokens_split_2  = ['[CLS]', 'Token', '##ize', 'word', '##piece', '.', '[SEP]']
        result_split_2 = str(combine_word_piece_vectors(vectors_split_2, tokens_split_2)[0])
        
        """
        print(result_split_2)
        print('[[3.4, 6.2, 4.5, -5.5], [3.8, 4.4, 4.95, 0.0], [5.4, -2.0, 4.5, 3.5]]')
        """

        vectors_split_2_plus = [[3.4, 6.2, 4.5, -5.5], [3.3, 6.2, 4.5, -5.5], [4.3, 2.6, 5.4, 5.5], [5.4, 3.2, 6.5, 8.5], [5.4, -7.2, 2.5, -1.5], [4.4, 2.2, 9.8, 7.4], [5.6, 7.6, 8.0, 2.2], [3.4, 6.2, 4.5, -5.5], [3.4, 6.2, 4.5, -5.5]]
        tokens_split_2_plus  = ['[CLS]', 'Token', '##ize', 'word', '##piece', 'extra', 'words', '.', '[SEP]']
        result_split_2_plus = str(combine_word_piece_vectors(vectors_split_2_plus, tokens_split_2_plus)[0])

        self.assertEqual(combine_word_piece_vectors(vectors_split_2, tokens_no_split)[0], vectors_no_split)    
        self.assertEqual(len(combine_word_piece_vectors(vectors_split_2, tokens_split_2)[0]), 3)        
        self.assertEqual(combine_word_piece_vectors(vectors_split_2, tokens_split_2)[1], 2)        
        self.assertEqual(result_split_2, '[[3.4, 6.2, 4.5, -5.5], [3.8, 4.4, 4.95, 0.0], [5.4, -2.0, 4.5, 3.5]]')
        self.assertEqual(result_split_2_plus, '[[3.4, 6.2, 4.5, -5.5], [3.8, 4.4, 4.95, 0.0], [5.4, -2.0, 4.5, 3.5], [4.4, 2.2, 9.8, 7.4], [5.6, 7.6, 8.0, 2.2]]')
        

    def test_get_ngram_embedding_vectors(self):
        
        model_name = 'bert-base-uncased'

        model, tokenizer = get_bert_model(model_name)

        embeddings, tokens = get_embeddings(['Test for wordpiece tokenizer and pad length.', 'Adding an additional sentence.'], model, 11, tokenizer)
        
        result_vectors_1 = get_ngram_embedding_vectors(embeddings, 1, True, tokens)
        result_vectors_2 = get_ngram_embedding_vectors(embeddings, 2, True, tokens)
        result_vectors_3 = get_ngram_embedding_vectors(embeddings, 3, True, tokens)
        result_vectors_1_no_pool = get_ngram_embedding_vectors(embeddings, 1, False, tokens)
        result_vectors_2_no_pool = get_ngram_embedding_vectors(embeddings, 2, False, tokens)
        result_vectors_3_no_pool = get_ngram_embedding_vectors(embeddings, 3, False, tokens)
        results_large_n = get_ngram_embedding_vectors(embeddings, 12, False, tokens)
        pooled_wp_2 = pool_vectors([embeddings[0][2] ,pool_vectors(embeddings[0][3:5])])
        pooled_wp_2_no_wp = pool_vectors(embeddings[0][2:4])
        pooled_wp_3 = pool_vectors( [embeddings[0][1], embeddings[0][2], pool_vectors(embeddings[0][3:5])])
        pooled_wp_3_no_wp = pool_vectors(embeddings[0][1:4])

        self.assertEqual(len(result_vectors_1), 11)
        self.assertEqual(len(result_vectors_2), 9)
        self.assertEqual(len(result_vectors_1_no_pool), 13)
        self.assertEqual(len(result_vectors_2_no_pool), 11)
        self.assertEqual(result_vectors_2[0], pool_vectors(embeddings[0][1:3]))
        self.assertEqual(result_vectors_2[1], pooled_wp_2)
        self.assertEqual(result_vectors_2_no_pool[1], pooled_wp_2_no_wp)
        self.assertEqual(len(result_vectors_3_no_pool), 9)
        self.assertEqual(len(result_vectors_3), 7)
        self.assertEqual(result_vectors_3[0], pooled_wp_3)
        self.assertEqual(result_vectors_3_no_pool[0], pooled_wp_3_no_wp)
        self.assertEqual(len(results_large_n), 2)


    def test_get_embedding_vectors(self):

        candidate_summaries = [ ['First candidate summary for testing.', 'Another sentence for testing purposes.', 'The final phrase is written here.', '.'], 
                                ['Second candidate summary is written here.', 'It only consists of two sentences.'], 
                                ['The third and final candidate summary is here.', 'It has more than two sentences.', 'Hence the third text sequence.'] 
                                ]
        
        reference_summaries = [ ['Here is the first sentence of the reference summary.', 'Only two individual sentences for this summary.'], 
                                ['Start of the second reference.', 'Testing the controlflow of the embedding functions.'], 
                                ['Lastly a single sentence reference summary.'] 
                                ]

        model, tokenizer = get_bert_model('deepset/sentence_bert')

        candidate_embeddings = []
        reference_embeddings = []

        for i, _ in enumerate(candidate_summaries, 0):
            cand_embs, ref_embs = get_embedding_vectors(candidate_summaries[i], reference_summaries[i], False, n_gram_encoding= None, layer = 11, model_name= 'deepset/sentence_bert', model= model, tokenizer = tokenizer)
            candidate_embeddings.append(cand_embs)
            reference_embeddings.append(ref_embs)


        self.assertEqual(len(candidate_embeddings), 3)
        self.assertEqual(len(candidate_embeddings[0]), 3)
        self.assertEqual(len(candidate_embeddings[1]), 2)
        self.assertEqual(len(candidate_embeddings[2]), 3)
        self.assertEqual(len(reference_embeddings), 3)
        self.assertEqual(len(reference_embeddings[0]), 2)
        self.assertEqual(len(reference_embeddings[1]), 2)
        self.assertEqual(len(reference_embeddings[2]), 1)
        
        model_name = 'bert-base-uncased'
        model_1, tokenizer_1 = get_bert_model(model_name)

        candidate_embeddings = []
        reference_embeddings = []

        for i, _ in enumerate(candidate_summaries, 0):
            cand_embs, ref_embs = get_embedding_vectors(candidate_summaries[i], reference_summaries[i], True, n_gram_encoding= 2, layer = 11, model_name= model_name, model= model_1, tokenizer= tokenizer_1)
            candidate_embeddings.append(cand_embs)
            reference_embeddings.append(ref_embs)

        self.assertEqual(len(candidate_embeddings), 3)
        self.assertEqual(len(candidate_embeddings[0]), 13)
        self.assertEqual(len(candidate_embeddings[1]), 10)
        self.assertEqual(len(candidate_embeddings[2]), 16)
        self.assertEqual(len(reference_embeddings), 3)
        self.assertEqual(len(reference_embeddings[0]), 14)
        self.assertEqual(len(reference_embeddings[1]), 10)
        self.assertEqual(len(reference_embeddings[2]), 5)