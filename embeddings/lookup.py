'''

    作用:根据词向量文件生成查找表
'''
import numpy as np
import pandas as pd

def generate(inFile,outFile):
    f = open(file=inFile, encoding="utf-8")
    lines = f.readlines()
    # first row is info
    info = lines[0].strip()
    info_list = info.split(sep=" ")
    vocab_size = int(info_list[0])
    embedding_dims = int(info_list[1])
    words = []
    ids = []
    for i in range(1, vocab_size + 1):
        embed = lines[i].strip()
        embed_list = embed.split(sep=" ")
        words.append(embed_list[0])
        ids.append(i)
    pd.DataFrame(data={"words": words, "id": ids}). \
        to_csv(path_or_buf=outFile, index=False, encoding="utf_8")

if __name__=="__main__":
    pass