import json
import os
import nltk
import re
from pymongo import MongoClient
from collections import defaultdict
from math import log, sqrt
from nltk.stem.snowball import EnglishStemmer

class BasicQuery:
    def __init__(self, directory, bookkeeping="bookkeeping.json"):
        self.client = MongoClient("localhost", 27017)
        self.db = self.client.search_engine             #search_engine is the name of the database
        self.collection = self.db.inverted_index        #inverted_index is the name of the specific collection
        self.bookkeeping = bookkeeping
        self.directory = directory
        self.stemmer = EnglishStemmer()
        self.total_docs = len(self.collection.distinct("docs.location"))


    """formats the urls with the complete protocol"""
    def format_urls(self, urls, protocol='http://'):
        full_urls = []
        for url in urls:
            if not url.startswith('http://') and not url.startswith('https://'):
                full_urls.append(f'{protocol}{url}')
            else:
                full_urls.append(url)
        return full_urls


    """returns a list of urls that match the query"""
    def query_index(self, query):
        results = self.calculate_cosine_similarity(query)
        urls = [self.get_url(doc_id) for doc_id in results] if results else []
        return self.format_urls(urls)


    """returns a list of urls associated with given doc_id"""
    def get_url(self, doc_id):
        file_path = os.path.join(self.directory, self.bookkeeping)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data[doc_id]
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file {file_path}")
            return None
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            return None


    """lemmatizes the query and returns a dictionary of lemmas : frequencies"""
    def tokenize_query(self, query):
        lemmas = defaultdict(int)

        content = re.sub('[^a-zA-Z0-9]', ' ', query)

        for lemma in [l.lower() for l in nltk.word_tokenize(content)]:
            if not (str(lemma).isnumeric()):
                lemma = self.stemmer.stem(lemma)
                lemmas[lemma] += 1

        return lemmas

    """calculates the normalized tf_idf for a query"""
    def process_query(self, query):
        query_tfidf = defaultdict(float)                    # dictionary containing query terms and corresponding tf_idf values
        lemmas = self.tokenize_query(query)              

        for lemma, tf in lemmas.items():                    # calculates the tf
            query_tfidf[lemma] = tf / len(lemmas)           # use the natural variant of tf

        for lemma in query_tfidf:                           # calculates doc frequency
            postings_list = self.collection.find_one({'lemma':lemma})['docs']       # log(total_docs / length of postings list)
            doc_freq = log(self.total_docs / len(postings_list)) if postings_list else 0
            query_tfidf[lemma] *= doc_freq

        vector_length = sqrt(sum(value ** 2 for value in query_tfidf.values()))     # normalize tf_idf values using Euclidean norm

        return query_tfidf, vector_length

    """calculates the length for each document vector"""
    def normalize_doc_tfidf(self, doc_vector):
        doc_vector_lengths = defaultdict(float)         # dictionary storing document_id and corresponding document vector length                                     

        for doc_id, lemma in doc_vector.items():        # iterate through each document vector
            doc_length = sqrt(sum(tf_idf ** 2 for tf_idf in lemma.values()))        # use Euclidean norm to calculate length
            doc_vector_lengths[doc_id] = doc_length                                 # store length in the dictionary

        return doc_vector_lengths

    """calculates cosine similarity between query and documents and returns
    the top 20 documents with the highest score"""
    def calculate_cosine_similarity(self, query):
        scores = defaultdict(float)                                     # stores cosine similarity scores
        query_tfidf, query_vector_length = self.process_query(query)    

        doc_vectors = defaultdict(lambda: defaultdict(list))            # structure is a nested dict: {doc_id : {lemma : list of tf_idf}}

        for lemma in query_tfidf.keys():                                # iterating through each lemma in the query
            postings_list = self.collection.find_one({'lemma': lemma})['docs']      # retrieve postings list

            for doc in postings_list:                                   # for each document in the postings list
                doc_id = doc['location']
                tf_idf = doc.get('tf_idf', doc['tf'] / log(self.total_docs / len(postings_list)))       # retrieve tf_idf value
                doc_vectors[doc_id][lemma] = tf_idf
                scores[doc_id] += doc['html_weight'] / 10000            # add html weight to score (/1000 for mininmal impact)

        self.normalize_doc_tfidf(doc_vectors)                           # normalize document scores

        for doc_id, data in doc_vectors.items():                        # for each document
            cosine_similarity = 0.0

            for lemma, tf_idf in data.items():                          # for each query lemma
                cosine_similarity += query_tfidf[lemma] * tf_idf        # accumulate cosine similarity score

            scores[doc_id] += cosine_similarity                         # accumulate total score (takes into account multiple term queries)

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)    # sort by accumulated cosine similarity score
        top_docs = [doc_id for doc_id, _ in sorted_scores[:20]]                     # return the doc_id for the 20 highest scores

        return top_docs