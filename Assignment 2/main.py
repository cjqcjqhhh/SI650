from os import system
from rankers import BM25Ranker, CustomRanker, PivotedLengthNormalizationRanker, Ranker
from pyserini.index import IndexReader
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import sys
import os


def get_score(ranker, query_id, query, dataset_id):
    '''
    Prints the relevance scores of the top retrieved documents.
    INPUT: ranker, a query
    OUTPUT: a list of top 10 tuples (score, docid) with highest score
    '''
    score_tuple_list = [] # list of (score, docid)
    print("------------------------------------")
    print("Start counting score for query: ", query_id, " ", query)
    for docid in tqdm(ranker.docid_list):
        score_tuple_list.append((ranker.score(query, docid), docid))

    # print the first n documents with highest ranking
    n = 5 if dataset_id == 2 or dataset_id == 3 else 10
    score_tuple_sorted = sorted(score_tuple_list, reverse = True)
    for score in score_tuple_sorted[:n]:
        print("docid=", score[1], "score=", score[0])
    return score_tuple_sorted[:n]

if __name__ == '__main__':

    if len(sys.argv) != 5:
        print("usage: python main.py dataset_id index_dirname query_filename ranker")
        exit(1)

    # arg1: reading the dataset_id
    if sys.argv[1] in ("1", "2", "3"):
        dataset_id = int(sys.argv[1])
    else:
        print(f"Error in dataset number {sys.argv[1]}, please choose an avaliable dataset number among 1, 2, or 3.")
        exit(1)
    dataset_dir = f"./data{dataset_id}/"

    # init the cache directory
    if not os.path.exists("./cache"):
        os.makedirs("./cache")

    # arg2: reading the index
    index_dir_path = dataset_dir + sys.argv[2]
    index_reader = IndexReader(index_dir_path)

    # arg3: reading the query
    query_df = pd.read_csv(dataset_dir + sys.argv[3])
    query_dict = query_df.set_index('QueryId').T.to_dict("records")[0] # key: queryId, value: Query Description
    if dataset_id == 2:
        sample_gaming_df = pd.read_csv(dataset_dir + "gaming_query_sample_submission.csv")
        sample_gaming_query_ids = set(sample_gaming_df["QueryId"])
        query_dict = {k: v for k, v in query_dict.items() if k in sample_gaming_query_ids}
    elif dataset_id == 3:
        sample_android_df = pd.read_csv(dataset_dir + "android_query_sample_submission.csv")
        sample_android_query_ids = set(sample_android_df["QueryId"])
        query_dict = {k: v for k, v in query_dict.items() if k in sample_android_query_ids}

    # arg4: choose the ranker
    if sys.argv[4] == "PLN":
        ranker = PivotedLengthNormalizationRanker(index_reader, dataset_id)
    elif sys.argv[4] == "BM25":
        ranker = BM25Ranker(index_reader, dataset_id)
    elif sys.argv[4] == "Custom":
        ranker = CustomRanker(index_reader, dataset_id)
    else:
        print(f"Error in ranker function {sys.argv[4]}: Please choose among available rankers: 'PLN', 'BM25', 'Custom'")

    # output the prediction result
    with open(f'output{dataset_id}.csv', 'w') as outfile:
        outfile.write('QueryId,DocumentId\n')
        for query_id, query in tqdm(query_dict.items()):
            res_score_tuple = get_score(ranker, query_id, query, dataset_id)
            print("Outputing prediction results...   Done. \n")
            for score, docid in res_score_tuple:
                outfile.write(str(query_id) + "," + str(docid) + '\n')

