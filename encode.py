import nltk
import numpy as np
import torch
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
from utils import bert_models


def pool_vectors(vectors):
    """
    Takes the average of the n vectors and returns the single result vector
    
    Args:
        - :param: 'vectors' (list of vectors): the embedding vectors to be combined
    Return:
        - :param: 'result_vector' (list of floats): the single output vector resulting from the combination
    """
    result_vector = np.mean(np.array(vectors), axis= 0)

    return result_vector.tolist()
    



def combine_word_piece_vectors(embedding_vectors, tokens):
    """
    Identifies the words that have been split by the BERT wordpiece tokenizer,
    and pools together their individual vectors into 1. 
    
    Args:
        - :param: 'embedding_vectors' (list of lists of floats): Word-embedding vectors for a single sentence
        - :param: 'tokens'            (list of str): list of the tokens, each corresponding to a embedding vector
    Return:
        - :param: 'pooled_wordpiece_vectors' (list of lists of floats): embeddings vectors post pooling of word-pieces
        - :param: 'valid_range' (int): index of the last vector in the matrix - used for the get_ngram_embedding_vectors function
    """
    
    pooled_wordpiece_vectors = [i for i in range(len(tokens))]
    valid_range = 0
    j = 0
    poolings = 0

    for i, token in enumerate(tokens, 0):
        if token.startswith('##'):
            pooled_wordpiece_vectors[j-1] = pool_vectors([embedding_vectors[i], pooled_wordpiece_vectors[j-1]])
            poolings += 1
        else:
            pooled_wordpiece_vectors[j] = embedding_vectors[i]
            j += 1
        
        valid_range = i - 2 - poolings 

    return pooled_wordpiece_vectors[:valid_range + 1], valid_range




def get_ngram_embedding_vectors(embedding_vectors, n_gram_encoding, pool_word_pieces, tokens):
    """
    Combines the word-level vectors into n-gram vectors.
    Ignores the vectors for [CLS], [SEP] and the final '.' 
    
    Args:
        - :param: `embedding_vectors` (list of list of lists of floats): embedding vectors for each token in each sentence in a summary -> Each sentence is represented as its own matrix
        - :param  'n_gram_encoding'   (int): n-gram encoding level - desginates how many word-vectors to combine for each final n-gram-embedding-vector            
                                             if 'None' -> Creates 1 vector pr. sentence in the summary                 
        - :param: `pool_word_pieces`  (bool): if True, pools together word-vectors for those words split by the wordpiece tokenizer
        - :param: `tokens`  (list of list of str): the individual tokens for each sentence - used for finding the valid range of vectors
    Return:
        - :param: - final_embeddings (list of list of floats): list of matricies of the embedding vectors for the summaries 
    """
    final_embeddings = []

    for i, sentence_matrix in enumerate(embedding_vectors, 0):
        valid_token_index = len(tokens[i]) - 3

        if valid_token_index <= 3: # Avoids sentences with 3 tokens or less  
            continue

        if n_gram_encoding is None:
            final_embeddings.append(pool_vectors(sentence_matrix[1:valid_token_index + 1]))
            continue

        if pool_word_pieces:
            sentence_matrix, valid_token_index = combine_word_piece_vectors(sentence_matrix, tokens[i])

        if n_gram_encoding >= valid_token_index: # Fewer or same amount of tokens as desired n-gram for pooling -> Defaults to making a single vector for the sentence
            final_embeddings.append(pool_vectors(sentence_matrix[1:valid_token_index + 1]))
            continue

        if n_gram_encoding == 1:
            vectors = sentence_matrix[1:valid_token_index + 1]
            for vec in vectors:
                final_embeddings.append(vec)
        
        else:
            n = 1 # Starting at position 1 to not include the [CLS] token 
            while n + n_gram_encoding <= valid_token_index + 1:
                end_index = n+n_gram_encoding
                final_embeddings.append(pool_vectors(sentence_matrix[n:end_index]))
                n += 1

    return final_embeddings       




def get_embeddings(summary, model, layer, tokenizer):
    """
    Retrieves the embeddings from the pretrained model using the transformers library
    Check README for details on the models supported for this

    Args:
        - :param: `summary`             (list of string): summary to be encoded
        - :param: `model`               (transformers model object): pretrained model from the transformers library to use of retrieving encodings
        - :param: `layer`               (int): the layer of representation to use
        - :param: `tokenizer`           (transformers tokenizer object): tokenizer for the specific model from the transformers library
    
    Return:
        - :param: final_embeddings (list of lists of lists of float): embedding vectors for each sentence in the summary
        - :param: final_tokens     (list of lists of string): list of the tokens in each sentence (special tokens not included)
                                                              Does not return tokens if the 'sentence-bert' model is used
    """

    final_tokens = []
    final_embeddings = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    model_inputs = tokenizer.batch_encode_plus(summary, max_length = 256)
    input_token_ids = model_inputs['input_ids']
    attention_masks = model_inputs['attention_mask']

    model.eval()
    with torch.no_grad():
        for i, _ in enumerate(input_token_ids, 0):
            vectors = []
            inputs =  torch.tensor([input_token_ids[i]])
            masks =  torch.tensor([attention_masks[i]])

            if device == 'cuda':
                inputs = inputs.to(device)
                masks = masks.to(device)

            hidden_states = model(inputs, masks)[2]
            vectors = hidden_states[layer][0].tolist()

            final_embeddings.append(vectors)

    for sentence in summary:
        tokens = tokenizer.tokenize(sentence)
        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
        final_tokens.append(tokens)

    return final_embeddings, final_tokens



def get_embedding_vectors(candidate_summary, reference_summary, pool_word_pieces, n_gram_encoding, layer, model_name, model, tokenizer = None):
    """
    Returns the embedding vectors for the candidate and reference summary 

    Args:
        - :param: `candidate_summaries` (list of list of strings): candidate summaries to be encoded - each summary should be represented as a list of sentences
        - :param: `reference_summaries` (list of list of strings): reference summaries to be encoded - each summary should be represented as a list of sentences
        - :param: `pool_word_pieces`    (bool): if True, pools together word-vectors for those words split by the wordpiece tokenizer 
        - :param: `layer`               (int): the layer of representation to use
        - :param: `model_name`          (str): the specific bert model to use
        - :param: `model`               (transformers model object): pretrained model from the transformers library to use of retrieving encodings
        - :param: `tokenizer`           (transformers tokenizer object): tokenizer for the specific model from the transformers library
        - :param  'n_gram_encoding'     (int): n-gram encoding level - desginates how many word vectors to combine for each final embedding vector
                                               if 'None' -> Generates a single vector pr. sentence in the summary 
    
    Return:
        - :param: candidate_embeddings, (list of lists of float): list of embedding vectors for the candidate summaries
        - :param: reference_embeddings, (list of lists of float): list of embedding vectors for the reference summaries
    """

    candidate_embeddings = []
    reference_embeddings = []

    cand_embeddings, cand_tokens = get_embeddings(candidate_summary, model, layer, tokenizer)
    ref_embeddings, ref_tokens = get_embeddings(reference_summary, model, layer, tokenizer)

    candidate_embeddings = get_ngram_embedding_vectors(cand_embeddings, n_gram_encoding, pool_word_pieces, cand_tokens)
    reference_embeddings = get_ngram_embedding_vectors(ref_embeddings, n_gram_encoding, pool_word_pieces, ref_tokens)

    return candidate_embeddings, reference_embeddings