import numpy as np
import pandas as pd
import os
import re
from gensim.models import word2vec


#原始语料转换为单个字的语料,可以训练字向量
def toCharCorpus(inFile,outFile):
    doc = ""
    file = open(file=inFile, encoding="utf-8")
    lines = file.readlines()
    # 每个字匹配一次
    pattern2 =re.compile(r"[^\s]")
    for line in lines:
        string=" ".join(re.findall(pattern=pattern2,string=line)) #每个字加上空格
        string+="\n"
        doc += string
    # write to file
    f = open(file=outFile, mode="w", encoding="utf-8")
    f.write(doc)
    f.close()



#训练词向量并且存储
def toCharEmbeddings(inFile):
    #--------------------------------train word embeddings---------------------------------
    sentences = word2vec.Text8Corpus(inFile)
    model = word2vec.Word2Vec(
        sentences=sentences,
        size=128,       # 词向量维度
        window=5,                       # window大小
        min_count=0,                    # 频率小于这个值被忽略
        sg=0,                           # sg==0->cbow;   sg==1->skip-gram
        hs=1,                           # use hierarchical softmax
        negative=10,                     # use negative sampling
        sorted_vocab=1,                 # 按照词频率从高到低排序
    )
    # save embeddings file
    if not os.path.exists("./embeddings"):
        os.mkdir(path="./embeddings")
    model.wv.save_word2vec_format("./embeddings/char_vec.txt", binary=False)

    # ----------------------------------生成char和id相互索引的.csv文件-------------------------
    if os.path.exists("./embeddings/char_vec.txt"):
        f = open(file="./embeddings/char_vec.txt", encoding="utf-8")
        lines = f.readlines()
        # first row is info
        info = lines[0].strip()
        info_list = info.split(sep=" ")
        vocab_size = int(info_list[0])
        embedding_dims = int(info_list[1])
        chars = []
        ids = []
        for i in range(1, vocab_size + 1):
            embed = lines[i].strip()
            embed_list = embed.split(sep=" ")
            chars.append(embed_list[0])
            ids.append(i)
        pd.DataFrame(data={"chars": chars, "id": ids}). \
            to_csv(path_or_buf="./embeddings/chars_ids.csv", index=False, encoding="utf_8")
    else:
        print("there is no embedings files")


if __name__ =="__main__":
    #先转为字符形式的
    toCharCorpus(inFile="./generate/final.txt",outFile="./generate/final_char.txt")
    toCharEmbeddings(inFile="./generate/final_char.txt")
    #toWordEmbeddings(inFile="./generate/final.txt")