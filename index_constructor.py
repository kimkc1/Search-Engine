import json
import os
import re
import nltk
import warnings
from bs4 import BeautifulSoup
from bs4 import MarkupResemblesLocatorWarning
from bs4 import XMLParsedAsHTMLWarning
from pymongo import UpdateOne
from collections import defaultdict
from basic_query import BasicQuery
from nltk.stem.snowball import EnglishStemmer
from pymongo.errors import BulkWriteError, CursorNotFound
from math import log
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio

warnings.filterwarnings("ignore", category = MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category = XMLParsedAsHTMLWarning)

"""use mongodb to store the inverted index
    https://pymongo.readthedocs.io/en/stable/tutorial.html
    run the following command: python3 -m pip install pymongo"""

class InvertedIndex:
    def __init__(self, html_dir, bookkeeping):
        self.html_dir = html_dir
        self.jf = os.path.join(".", self.html_dir, "bookkeeping.json")
        self.bookkeeping = bookkeeping
        self.num_documents = 0
        self.file = 'analytics.txt'
        self.client = AsyncIOMotorClient("localhost", 27017)
        self.db = self.client.search_engine
        self.collection = self.db.inverted_index
        self.stemmer = EnglishStemmer()
        self.stopWords = set(line.strip() for line in open('stop_words.txt'))
        self.htmlWeights = {'title':0.6, 'h1':0.5, 'h2':0.4, 'h3':0.3, 'h4':0.2}


    """iterates through the corpus and processes each document"""
    async def build_index(self):
        if self.collection.count_documents({}) == 0:    # checks if the index is already built
            print("Index is empty. Building index...")
            big_dir = json.load(open(self.jf, encoding = "utf-8"))
            semaphore = asyncio.Semaphore(3)            # max of 3 concurrent tasks

            async def process_dir(directory):
                async with semaphore:
                    folder, file = directory.split("/")
                    file_path = os.path.join(".", self.html_dir, folder, file)
                    soup = BeautifulSoup(open(file_path, 'r', encoding = 'utf-8'), 'lxml')

                    lemmas = self.process_document(soup)                # lemmas is a dict containing lemma, freq, html_weight
                    await self.add_to_index(lemmas, folder, file)

            tasks = [process_dir(directory) for directory in big_dir]   # create a task for each directory and run them concurrently

            await asyncio.gather(*tasks)            # makes sure that all tasks have been completed before continuing
            await self.calculate_tf_idf()           # adds tf_idf values to the database
            await self.generate_analytics()         # generates the analytics for milestone 1


    """generates a dictionary containing lemmas, frequencies, and html weights for each document"""
    def process_document(self, soup):
        lemmas = defaultdict(lambda: {'freq': 0, 'html_weight': 0})        # stores frequency and html weight

        for tag in soup.find_all(True):                                    # retrieves all tags in the html document
            tag_name = tag.name                                            # tag.name gets the html tag type (ex. h1, h2, etc.)
            html_weight = self.htmlWeights.get(tag_name, 0.1)              # assign weight depending on tag, default to 1
            content = re.sub('[^a-zA-Z0-9]', ' ', tag.text)                # extracts the actual textual content

            for lemma in [l.lower() for l in nltk.word_tokenize(content) if (not l.lower() in self.stopWords) and len(l) > 2]:
                if not (str(lemma).isnumeric()):
                    lemma = self.stemmer.stem(lemma)
                    lemmas[lemma]['freq'] += 1                           # update frequency of each lemma
                    lemmas[lemma]['html_weight'] += html_weight          # update html weight of each lemma

        return lemmas


    """writes each document's data to the mongo database"""
    async def add_to_index(self, lemmas, folder, file):
        total_words = len(lemmas)
        updates = []

        print(f'Adding {folder}/{file} to DB ({self.num_documents})')
        for lemma, data in lemmas.items():                                                      # adds each lemma to the database
            doc_id = f"{folder}/{file}"
            tf = data['freq'] / total_words
            html_weight = data['html_weight']
            doc_entry = {'location': doc_id, 'tf': tf, 'html_weight': html_weight}
            updates.append(
                UpdateOne({'lemma': lemma}, {'$push': {'docs': doc_entry}}, upsert = True))     # use bulk updating to reduce writing time

        if updates:  # check if the updates list is not empty
            try:
                await asyncio.wait_for(self.collection.bulk_write(updates), timeout=20)
                self.num_documents += 1
            except asyncio.TimeoutError:
                print("The operation timed out")
            except BulkWriteError as e:
                print("Error adding lemmas from directory:", doc_id)
            except Exception as e:
                print(f"Write operation error: {e}")
            except KeyboardInterrupt:
                raise KeyboardInterrupt


    """calculates the tf_idf for each term/doc given the database"""
    # use a cursor to iterate through database (documentation: https://www.mongodb.com/docs/manual/tutorial/iterate-a-cursor/)
    async def calculate_tf_idf(self):
        total_docs = len(self.collection.distinct("docs.location"))
        cursor = self.collection.find()

        async for entry in cursor:                          # iterates through each entry in the database
            lemma = entry['lemma']
            df = len(entry['docs'])
            idf = log(total_docs / df) if df > 0 else 0
            updates = []

            for doc in entry['docs']:                       # iterates through each document in the postings list for a lemma
                location = doc['location']
                tf_idf = doc['tf'] * idf                    # uses previously stored tf values and calculated idf for each lemma
                print(f"Calculated tf_idf for {lemma}, {location}: {tf_idf}")

                updates.append(                             # updates database to include tf_idf values
                    UpdateOne(
                        {'lemma': lemma, 'docs.location': location},
                        {'$set': {'docs.$.tf_idf': tf_idf}}, upsert = True))
                
            try:
                self.collection.bulk_write(updates)
            except Exception as e:
                print("Error calculating tf_idf:", e)

        print("tf_idf calculations complete.")


    """generates analytics for milestone 1"""
    async def generate_analytics(self):
        # documentation on dbStats: https://www.mongodb.com/docs/manual/reference/command/dbStats/
        # documentation on distinct: https://www.mongodb.com/docs/manual/reference/method/db.collection.distinct/
        
        db_stats = await self.client.admin.command('dbStats')
        db_size = db_stats['dataSize'] / 1024
        unique_words = await self.collection.distinct('lemma')

        test_queries = ['informatics', 'mondego', 'irvine']     # generate queries for 3 words (Informatics, Mondego, Irvine)
        query_engine = BasicQuery(self.html_dir)

        with open(self.file, 'w') as f:
            print("running analytics now\n")
            f.write(f"Analytics for Milestone #1\n\n")
            f.write(f"Total number of documents: {self.num_documents}\n\n")
            f.write(f"Total size of index on disk: {db_size} KB\n\n")
            f.write(f"Number of unique words: {len(unique_words)}\n\n")
            print("testing queries now")
            f.write(f"Testing Queries\n")
            for query in test_queries:
                results = query_engine.query_index(query)
                f.write(f"Query: {query}\n")
                f.write(f"Number of URLs retrieved: {len(results)}\n")
                f.write(f"First 20 URLs:\n")
                for url in results[:20]:
                    f.write(f"{url}\n")
                f.write("\n")