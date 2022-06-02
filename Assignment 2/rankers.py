from pyserini.index import IndexReader
from tqdm import tqdm, trange
import sys
import numpy as np
import json


class Ranker(object):
    '''
    The base class for ranking functions. Specific ranking functions should
    extend the score() function, which returns the relevance of a particular 
    document for a given query.
    '''
    
    def __init__(self, index_reader, dataset_id):
        self.index_reader = index_reader
        self.dataset_id = dataset_id
        self.doc_num = index_reader.stats()["documents"] # number of document
        self.tot_doc_len = 0 # total document byte length
        self.docid_list = [] # list of unique document id
        self.doc_dict = {} # dict of analyzed document vector 
                           # key: docid, value: term_freq_dict for document vector
        self.term_dict = {} # dict of term freq in whole collection
                            # key: term, value: df or (df, cf)

        print("------- Init the ranker: --------")
        print(f"---> Using dataset{self.dataset_id}...")
        # cache a list of document id
        print("Init docid_list:")
        for i in trange(self.doc_num):
            docid = index_reader.convert_internal_docid_to_collection_docid(i)
            self.docid_list.append(docid)
        # cache dict of document vector
        print("Init doc_dict:")
        try:
            with open(f"./cache/doc_dict{self.dataset_id}.json", "r") as infile:
                self.doc_dict = json.load(infile)
            print("<--- Read from cache.")
        except:
            for docid in tqdm(self.docid_list):
                self.doc_dict[docid] = self.index_reader.get_document_vector(docid)
            with open(f"./cache/doc_dict{self.dataset_id}.json", "w") as outfile:
                json.dump(self.doc_dict, outfile, indent = 4)
            print("---> Successfully caching doc_dict!")
        # count the total document length
        self.tot_doc_len = sum([len(self.index_reader.doc_contents(docid)) for docid in self.docid_list])

    def score(query, doc):        
        '''
        Returns the score for how relevant this document is to the provided query.
        Query is a tokenized list of query terms and doc_id is the identifier
        of the document in the index should be scored for this query.
        '''
        
        rank_score = 0
        return rank_score


class PivotedLengthNormalizationRanker(Ranker):
    
    def __init__(self, index_reader, dataset_id):
        super(PivotedLengthNormalizationRanker, self).__init__(index_reader, dataset_id)
        # cache the term document freq
        print("Init term_dict:")
        try:
            with open(f"./cache/term_dict{self.dataset_id}.json", "r") as infile:
                self.term_dict = json.load(infile)
            print("<--- Read from cache.")
        except:
            for term in tqdm(index_reader.terms()):
                self.term_dict[term.term] = term.df
            with open(f"./cache/term_dict{self.dataset_id}.json", "w") as outfile:
                json.dump(self.term_dict, outfile, indent = 4)
            print("---> Successfully caching term_dict!")
        
    def score(self, query, docid, b = 0.15):
        '''
        Scores the relevance of the document for the provided query using the
        Pivoted Length Normalization ranking method. Query is a tokenized list
        of query terms and doc_id is a numeric identifier of which document in the
        index should be scored for this query.
        '''

        rank_score = 0
        # get necessary values:
        doc_len = len(self.index_reader.doc_contents(docid)) # byte len of document: |d|
        avg_doc_len = self.tot_doc_len / self.doc_num # average byte len of document: avg_dl
        doc_analyzed = self.doc_dict[docid] # analyzed documnet dict - key: term, value: freq
                                            # doc_analyzed[term]: c(w,d)
        query_analyzed = self.index_reader.analyze(query)
        query_term_dict = {t: query_analyzed.count(t) for t in query_analyzed} # key: term, value: freq

        # Count the score
        for term, freq in query_term_dict.items():
            try:
                doc_freq = self.term_dict[term] # df
                if doc_freq > 0 and term in doc_analyzed:
                    qtf = freq
                    tf = (1 + np.log(1 + np.log(doc_analyzed[term]))) / (1 - b + b * doc_len / avg_doc_len)
                    idf = np.log((self.doc_num + 1) / doc_freq)
                    rank_score += qtf * tf * idf
            except:
                pass

        return rank_score

    

class BM25Ranker(Ranker):

    def __init__(self, index_reader, dataset_id):
        super(BM25Ranker, self).__init__(index_reader, dataset_id)
        # cache the term document freq
        print("Init term_dict:")
        try:
            with open(f"./cache/term_dict{self.dataset_id}.json", "r") as infile:
                self.term_dict = json.load(infile)
            print("<--- Read from cache.")
        except:
            for term in tqdm(index_reader.terms()):
                self.term_dict[term.term] = term.df
            with open(f"./cache/term_dict{self.dataset_id}.json", "w") as outfile:
                json.dump(self.term_dict, outfile, indent = 4)
            print("---> Successfully caching term_dict!")
        
    def score(self, query, docid, k1=1.2, b=0.75, k3=1.25):
        '''
        Scores the relevance of the document for the provided query using the
        BM25 ranking method. Query is a tokenized list of query terms and doc_id
        is a numeric identifier of which document in the index should be scored
        for this query.
        '''

        rank_score = 0
        if docid not in self.doc_dict.keys():
            return 0

        doc_len = len(self.index_reader.doc_contents(docid))
        avg_doc_len = self.tot_doc_len / self.doc_num # average byte len of document: avg_dl
        doc_analyzed = self.doc_dict[docid] # analyzed documnet dict
                                            # doc_analyzed[term]: c(w,d)
        query_analyzed = self.index_reader.analyze(query)
        query_term_dict = {t: query_analyzed.count(t) for t in query_analyzed} # key: term, value: freq of term in query
        
        for term, freq in query_term_dict.items():
            doc_freq = self.term_dict[term] if term in self.term_dict.keys() else 0 # df
            if doc_freq > 0 and term in doc_analyzed:
                idf = np.log((self.doc_num - doc_freq + 0.5) / (doc_freq + 0.5))
                tf = (k1 + 1) * doc_analyzed[term] / (k1 * (1 - b + b * doc_len / avg_doc_len) + doc_analyzed[term])
                qtf = (k3 + 1) * freq / (k3 + freq)
                rank_score += idf * tf * qtf

        return rank_score



    
class CustomRanker(Ranker):
    
    def __init__(self, index_reader, dataset_id):
        super(CustomRanker, self).__init__(index_reader, dataset_id)
        self.term_dict = {} # dict of term freq in whole collection
                               # key: term, value: df, cf
        # cache the term collection freq
        print("Init term_dict:")
        try:
            with open(f"./cache/term_dict{self.dataset_id}.json", "r") as infile:
                self.term_dict = json.load(infile)
            print("<--- Read from cache.")
        except:
            for term in tqdm(index_reader.terms()):
                self.term_dict[term.term] = (term.df, term.cf)
            with open(f"./cache/term_dict{self.dataset_id}.json", "w") as outfile:
                json.dump(self.term_dict, outfile, indent = 4)
            print("---> Successfully caching term_dict!")
        # total number of terms in collection
        self.term_num = sum([len(term) * freq[1] for term, freq in self.term_dict.items()])
        

    def score(self, query, docid, k1=1.2, b=0.75, k2=1.2, k3=1/100):
        '''
        Scores the relevance of the document for the provided query using a
        custom ranking method. Query is a tokenized list of query terms and doc_id
        is a numeric identifier of which document in the index should be scored
        for this query.
        '''

        rank_score = 0
        if docid not in self.doc_dict.keys():
            return 0

        doc_len = len(self.index_reader.doc_contents(docid))
        avg_doc_len = self.tot_doc_len / self.doc_num # average byte len of document: avg_dl
        doc_analyzed = self.doc_dict[docid] # analyzed documnet dict
                                            # doc_analyzed[term]: c(w,d)
        query_analyzed = self.index_reader.analyze(query)
        query_term_dict = {t: query_analyzed.count(t) for t in query_analyzed} # key: term, value: freq of term in query
        
        
        for term, freq in query_term_dict.items():
            doc_freq = self.term_dict[term][0] if term in self.term_dict.keys() else 0 # df
            clt_freq = self.term_dict[term][1] if term in self.term_dict.keys() else 0 # cf
            if doc_freq > 0 and term in doc_analyzed:
                idf = np.log((self.doc_num + 1) / (doc_freq + 1))
                tf = (k1 + 1) * doc_analyzed[term] / (k1 * (1 - b + b * doc_len / avg_doc_len) + doc_analyzed[term])
                qtf = freq
                cf = k3 / (k3 + clt_freq / self.term_num) # a normalized cf
                rank_score += idf * tf * qtf * cf

        return rank_score