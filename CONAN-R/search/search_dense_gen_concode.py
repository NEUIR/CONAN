# This file is .....
# Author: Hanbin Wang
# Date: 2023/4/15
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import faiss
import logging
logger = logging.getLogger(__name__)

def sum_load_data(query_file,corpus_file):
    queries = []
    golds = []
    #load query file
    with open(query_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            line = json.loads(line)
            query =line['nl']
            gold =line['code']
            queries.append(query)
            golds.append(gold)
        logger.info(f'   load queries and golds from file {query_file}')

    # load corpus file
    corpus = []
    with open(corpus_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            line = json.loads(line)
            corpus.append(line)
        logger.info(f"  load corpus from file {corpus_file}")

    return queries,golds,corpus


def search(args,index_file, query_file, save_name,queries_text,golds,corpus):
    logger.warning("    search!")
    index_data = pickle.load(open(index_file, "rb"))
    query_data = pickle.load(open(query_file, "rb"))
    ids = []
    indexs = []
    # 记录corpus的索引和768维向量
    for i, (idx, vec) in enumerate(index_data.items()):
        ids.append(idx)
        indexs.append(vec)

    # 记录queries的id和768维向量
    queries = []
    idxq = []
    for idx, vec in query_data.items():
        queries.append(vec)
        idxq.append(idx)


    ids = np.array(ids)
    indexs = np.array(indexs)

    # 归一化向量
    # indexs /= np.linalg.norm(indexs, axis=1, keepdims=True)

    queries = np.array(queries)

    # 归一化查询向量
    # queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # 设置温度
    # temperature = 0.02

    # build faiss index
    d = 768
    k = 100
    index = faiss.IndexFlatIP(d)
    assert index.is_trained

    index_id = faiss.IndexIDMap(index)
    index_id.add_with_ids(indexs, ids)

    search_result = []
    D, I = index_id.search(queries, k) #返回的结果是最相似的k个嵌入向量的索引和它们与查询向量的相似性得分。D 和 I 分别指的是距离和索引
    # 将得分除以温度
    # D =D / temperature

    for i, (sd, si) in enumerate(zip(D, I)):
        # 第i个query的结果，str(idxq[i]表示第i个query的id
        res = {}
        # query
        res['id'] = i
        res['question']=queries_text[i]
        # target
        res['target'] = golds[i]
        res['answers']=golds[i]
        # 检索结果，top1，top2...
        res['ctxs']=[
            {
                'id':args.corpus_file + '_' + str(pi),
                'title':"",
                'text':corpus[pi]['code'],
                'linked':corpus[pi]['nl'],
                'score':pd,
                'has_answer':True
            }for pd, pi in zip(sd, si)
        ]
        search_result.append(res)

    pickle.dump(search_result, open(save_name, "wb"))
    logger.warning("    search done!")



def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--index_vector_file', '-i', required=True, help="filename of index embeddings saved")
    parser.add_argument('--query_vector_file', '-q', required=True, help="file containing query embeddings")
    parser.add_argument('--query_file', '-qf', required=True, help="file containing original query text")
    parser.add_argument('--corpus_file', '-cf', required=True, help="file containing original corpus text")
    parser.add_argument('--save_name', '-o', required=True, help="save file name")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    queries, golds, corpus = sum_load_data(args.query_file,args.corpus_file)
    search(args,args.index_vector_file, args.query_vector_file, args.save_name,queries,golds,corpus)

if __name__ == "__main__":
    main()

