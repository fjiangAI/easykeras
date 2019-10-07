__author__ = 'jf'
from easykeras.keras_bert.tokenizer import Tokenizer
from easykeras.keras_bert import load_vocabulary, get_checkpoint_paths, load_trained_model_from_checkpoint
from easykeras.keras_bert.util import extract_embeddings,extract_embeddings_bymodel
from keras.layers import *


def bert_embedding(model_path, texts_seq):
    embeddings_seq = extract_embeddings(model_path, texts_seq)
    return embeddings_seq


def bert_embedding_by_model(seq_len, tokenizer, model, texts_seq):
    embeddings_seq=extract_embeddings_bymodel(seq_len, tokenizer, model, texts_seq)
    return embeddings_seq


def get_model_and_config(model_path, cased=False):
    """
    获取BERT的配置
    :param model_path:
    :param cased:
    :return:
    """
    paths = get_checkpoint_paths(model_path)
    model = load_trained_model_from_checkpoint(
            config_file=paths.config,
            checkpoint_file=paths.checkpoint
        )
    vocabs = load_vocabulary(paths.vocab)
    seq_len = K.int_shape(model.outputs[0])[1]
    tokenizer = Tokenizer(vocabs, cased=cased)
    return seq_len, tokenizer, model