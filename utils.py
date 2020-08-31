import torch
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel


# In order to setup the Danish BERT model properly please consult the repo README
# deepset sentence model is an instance of the 'bert-base-nli-stsb-mean-tokens' pretrained model from the Sentence Transformers Repo (https://github.com/UKPLab/sentence-transformers)
bert_models = ['bert-base-uncased',
               'bert-base-cased',
               'bert-large-uncased',
               'bert-large-cased',
               'bert-base-multilingual-uncased',
               'bert-base-multilingual-cased',
               'danish-bert',                  
               'deepset/sentence_bert'
               ]


roberta_models = ['roberta-base',
                  'roberta-large']

def get_bert_model(model_name):
    """
    Retrieves the pretrained model and tokenizer from the transformers library.
    """
    model = None
    tokenizer = None

    if model_name == 'danish-bert':                    
        model_directory = 'C:/Users/Lukas/ITU/Master_Thesis/Transformers/bert/bert-base-danish-uncased-v2'
        tokenizer = BertTokenizer.from_pretrained(model_directory)
        model = BertModel.from_pretrained(model_directory, output_hidden_states = True)

    elif model_name in bert_models:
        tokenizer = BertTokenizer.from_pretrained(model_name)        
        model = BertModel.from_pretrained(model_name, output_hidden_states=True)

    else: 
        print('model must be specified as one of the supported ones. Check readme for more details')
        return

    return model, tokenizer



# Deprecated lists and functions related to bert-as-service functionality 
"""
from bert_serving.client import BertClient
from bert_serving.server.helper import get_args_parser, get_shutdown_parser
from bert_serving.server import BertServer
"""

bert_model_directories = {'bert-base-uncased' : 'C:/Users/Lukas/ITU/Master_Thesis/Transformers/bert/uncased_L-12_H-768_A-12/', 
                          'bert-base-cased' : 'C:/Users/Lukas/ITU/Master_Thesis/Transformers/bert/cased_L-12_H-768_A-12/', 
                          'mbert' : 'C:/Users/Lukas/ITU/Master_Thesis/Transformers/bert/multi_cased_L-12_H-768_A-12/'}


# Convention of layer naming for the bert-as-service server
layers_bert_base = {1 : '-12', 
                 2 : '-11', 
                 3 : '-10', 
                 4 : '-9',
                 5 : '-8',
                 6 : '-7',
                 7 : '-6',
                 8 : '-5', 
                 9 : '-4',
                 10 : '-3',
                 11 : '-2',
                 12 : '-1'}

pooling_strategies = ['REDUCE_MEAN', 
                      'REDUCE_MAX', 
                      'REDUCE_MEAN_MAX',
                      'CLS_TOKEN',
                      'SEP_TOKEN']


def launch_bert_as_service_server(model_name, layer, encoding_level = None, pooling_strategy = None):
    """
    Launches a BERT-as-service server used to encode the sentences using the designated BERT model
    https://github.com/hanxiao/bert-as-service

    Args:
        - :param: `model_name`       (str): the specific bert model to use
        - :param: `layer`            (int): the layer of representation to use
        - :param  'encoding_level'   (int): gram encoding level - desginates how many word vectors to combine for each final embedding vector
                                            if 'None' -> embedding level defaults to the sentence level of each individual sentence
        - :param: `pooling_strategy` (str): the vector combination strategy - used when 'encoding_level' == 'sentence' 
    """

    model_path = bert_model_directories[model_name]
    pooling_layer = layers_bert_base[layer]

    server_parameters = ""
        
    if encoding_level == None:
        
        if pooling_strategy not in pooling_strategies:
            print('"pooling_strategy" must be defined as one of the following:', pooling_strategies)
            return

        server_parameters = get_args_parser().parse_args(['-model_dir', model_path,
                                        '-port', '5555',
                                        '-port_out', '5556',
                                        '-max_seq_len', '100',                                        
                                        '-pooling_layer', pooling_layer,
                                        '-pooling_strategy', pooling_strategy, 
                                        '-num_worker=1'])
    
    
    elif encoding_level >=1:
                server_parameters = get_args_parser().parse_args(['-model_dir', model_path,
                                        '-port', '5555',
                                        '-port_out', '5556',
                                        '-max_seq_len', '100',                                        
                                        '-pooling_layer', pooling_layer,
                                        '-pooling_strategy', 'NONE',
                                        '-show_tokens_to_client',
                                        '-num_worker=1'])
    else:
        print('"encoding_level" must be >=1 or None, see README for descriptions')
        return

    server = BertServer(server_parameters)
    print("LAUNCHING SERVER, PLEASE HOLD", '\n')
    server.start()
    print("SERVER RUNNING, BEGGINING ENCODING...")



def terminate_server():
    print("ENCODINGS COMPLETED, TERMINATING SERVER...")
    shutdown = get_shutdown_parser().parse_args(['-ip','localhost','-port','5555','-timeout','5000'])
    BertServer.shutdown(shutdown)