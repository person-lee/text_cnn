# coding=utf-8

import codecs
import logging
import os
import numpy as np
from collections import defaultdict

#----------------------------- define a logger ---------------------------------------
logging.basicConfig(format="%(message)s", level=logging.INFO)
#----------------------------- define a logger end -----------------------------------
def build_vocab(filename):
    word2idx, idx2word = defaultdict(), defaultdict()
    try:
        with codecs.open(filename, mode="r", encoding="utf-8") as rf:
            for line in rf.readlines():
                items = line.strip().split(" ")
                if len(items) != 2:
                    continue
                word_id = int(items[1].strip()) - 1
                word = items[0].strip()
                idx2word[word_id] = word
                word2idx[word] = word_id

            logging.info("build_vocab finish")
            rf.close()

        word2idx["UNKNOWN"] = len(idx2word)
        idx2word[len(idx2word)] = "UNKNOWN"
        word2idx["<a>"] = len(idx2word)
        idx2word[len(idx2word)] = "<a>"
    except Exception, e:
        logging.info(e)
    return word2idx, idx2word


def load_embedding(embedding_size, word2idx=None, filename=None):
    if filename is None and word2idx is not None:
        return load_embedding_random_init(word2idx, embedding_size)
    else:
        return load_embedding_from_file(filename)

def load_embedding_random_init(word2idx, embedding_size):
    embeddings=[]
    for word, idx in word2idx.items():
        vec = [0.01 for i in range(embedding_size)]
        embeddings.append(vec)
    return np.array(embeddings, dtype="float32")

def load_embedding_from_file(embedding_file):
    word2vec_embeddings = np.array([ [float(v) for v in line.strip().split(' ')] for line in open(embedding_file).readlines()], dtype=np.float32)
    embedding_size = word2vec_embeddings.shape[1]
    unknown_padding_embedding = np.random.normal(0, 0.1, (2,embedding_size))

    embeddings = np.append(word2vec_embeddings, unknown_padding_embedding.astype(np.float32), axis=0)
    return embeddings


def load_data(filename, word2idx, sequence_len):
    label2sents = defaultdict()
    unknown_id = word2idx.get("UNKNOWN", 0)
    try:
        with codecs.open(filename, mode="r", encoding="utf-8") as rf:
            for line in rf.readlines():
                items = line.strip().split(" ")
                label = items[-1].strip()
                sentences = label2sents.get(label, [])
                sent_idx = [word2idx.get(word.strip(), unknown_id) for word in items[:-1]]

                # padding
                pad_idx = word2idx.get("<a>", 0)
                if len(sent_idx) < sequence_len:
                    pad_num = sequence_len - len(sent_idx)
                    for i in range(pad_num):
                        sent_idx.append(pad_idx) 
                else:
                    sent_idx = sent_idx[:sequence_len]
                sentences.append(sent_idx)
                label2sents[label] = sentences

        rf.close()
    except Exception, e:
        logging.info(e)

    return label2sents

def create_valid(data, proportion=0.1):
    if data is None:
        logging.error("data is none")
        os._exit(1)
    data_len = len(data)
    seperate_idx = int(data_len * (1 - proportion))
    return data[:seperate_idx], data[seperate_idx:]

def batch_iter(data, batch_size, epoches, shuffle=True):
    """
    iterate the data
    """
    data_len = len(data)
    batch_num = int(data_len / batch_size) + 1
    data = np.array(data)

    for epoch in range(epoches):
        if shuffle:
            shuffle_idx = np.random.permutation(np.arange(data_len))
            shuffle_data = data[shuffle_idx]
        else:
            shuffle_data = data

        for batch in range(batch_num):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, data_len)
            yield shuffle_data[start_idx : end_idx]
