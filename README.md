# Text Similarity Assesment using BERT (TSAuBERT)

Using the contextually-dependent embeddings obtained from BERT to assess the similarity of a candidate- and reference-text using cosine similarity. 
The main application of this metric is to assess the similarity between a candidate- and reference-summary.

### Algorithm
The scoring algorithm consists of two central steps:

#### 1) Encoding 
Obtaining embedding vectors from a pretrained BERT-based model. 

#### 2) Scoring 
Calculating the score using cosine similarity. 


### Installation requirements
* Python version >= 3.6
* huggingface/transformers (https://github.com/huggingface/transformers)
* nltk (packages for tokenization for different languages)
* for Danish BERT model setup, follow the guide created by Daniel Varab Here: https://github.com/danielvarab/convert_da_bert
