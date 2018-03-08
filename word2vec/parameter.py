
IN_FILE="./all_test.txt"            #corpus File
OUT_FOLDER="./embeddings/"          #Folder put the generated word2vec file and lookup table

VECTOR_DIM=128
WINDOW_SIZE=5
MIN_COUNT=0
TYPE=0                              # sg==0->cbow;   sg==1->skip-gram
HS=1                                # use hierarchical softmax
NEGATIVE=10                         # use negative sampling
SORT=1                              # sort by word frequncy


