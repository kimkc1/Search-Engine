import sys
from index_constructor import InvertedIndex
from basic_query import BasicQuery
from GUI import start_GUI
from concurrent.futures import ThreadPoolExecutor
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio

"""command to run the program:
    python main.py /Users/kaylakim/Desktop/WEBPAGES_RAW /Users/kaylakim/Desktop/WEBPAGES_RAW/bookkeeping.json
    python main.py (path to corpus) (path to webpages raw)"""

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide path to corpus")
        sys.exit(1)

    corpus_path = sys.argv[1]
    bookkeeping_path = sys.argv[2]
    index = InvertedIndex(corpus_path, bookkeeping_path)    # creating an InvertedIndex object

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(index.build_index())            # builds index if not previously built

    query = BasicQuery(corpus_path)                         # creating a BasicQuery object
    start_GUI(query)                                        # starts the GUI

    loop.close()

