import numpy as np
import pandas as pd
import os
import time
from gensim.models import word2vec
import parameter
import lookup

#训练词向量并且存储
def toWordEmbeddings():
    #--------------------------------train word embeddings---------------------------------
    print("training word embeddings.....")
    sentences = word2vec.Text8Corpus(parameter.IN_FILE)
    model = word2vec.Word2Vec(
        sentences=sentences,
        size=parameter.VECTOR_DIM,          # 词向量维度
        window=parameter.WINDOW_SIZE,       # window大小
        min_count=parameter.MIN_COUNT,      # 频率小于这个值被忽略
        sg=parameter.TYPE,                  # sg==0->cbow;   sg==1->skip-gram
        hs=parameter.HS,                    # use hierarchical softmax
        negative=parameter.NEGATIVE,        # use negative sampling
        sorted_vocab=parameter.SORT,        # 按照词频率从高到低排序
    )
    # save embeddings file
    if not os.path.exists(parameter.OUT_FOLDER):
        os.mkdir(path=parameter.OUT_FOLDER)
    model.wv.save_word2vec_format(parameter.OUT_FOLDER+"word_vec.txt", binary=False)

    # ----------------------------------生成word和id相互索引的.csv文件-------------------------
    print("generating lookup table.....")
    if os.path.exists(parameter.OUT_FOLDER+"word_vec.txt"):
        lookup.generate(inFile=parameter.OUT_FOLDER+"word_vec.txt",outFile=parameter.OUT_FOLDER+"words_ids.csv")
    else:
        print("there is no embedings files")


if __name__ =="__main__":
    begin_time=time.time()
    toWordEmbeddings()
    end_time=time.time()
    print("ALL DONE!")
    print("Spend ",(end_time-begin_time)/60," mins")
