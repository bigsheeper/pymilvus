import random
import threading
import time
import gc
import numpy as np
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from config import TopK, NQ, Nprobe, NumberOfTestRun, collection_name, field_name, dim, nb, batch, thread_nums, ef

connections.connect("default")

from pymilvus.orm.types import CONSISTENCY_EVENTUALLY
from pymilvus.orm.types import CONSISTENCY_BOUNDED

def time_costing(func):
    def core(*args):
        start = time.time()
        res = func(*args)
        end = time.time()
        print(func.__name__, "time cost: ", end-start)
        return res
    return core

# @time_costing
def search(collection, query_entities, field_name, topK, nprobe):
    search_params = {"metric_type": "L2", "params": {"ef": ef}}
    res = collection.search(query_entities, field_name, search_params, limit=topK, consistency_level=CONSISTENCY_EVENTUALLY)
    # res = collection.search(query_entities, field_name, search_params, limit=topK)


def generate_entities(dim, nb) -> list:
    vectors = [[random.random() for _ in range(dim)] for _ in range(nb)]
    return vectors

if __name__ == "__main__":
    coll = Collection(collection_name)

    coll.release()
    coll.load()

    print("topK nq avg_latency/{:d}".format(NumberOfTestRun))
    for topK in TopK:
        for nq in NQ:
            for nprobe in Nprobe:
                # print("nprobe = ", nprobe, "topK = ", topK, "nq = ", nq)
                start = time.time()
                for _ in range(NumberOfTestRun):
                    query_entities = generate_entities(dim, nq)
                    search(coll, query_entities, field_name, topK, nprobe)

                end = time.time()
                # print("nprobe = ", nprobe, "topK = ", topK, "nq = ", nq, "test times = ", NumberOfTestRun, "total time = ",
                #       end - start, "avg time = ", (end-start)/NumberOfTestRun)
                # print("topK = ", topK, "nq = ", nq, "times = ", NumberOfTestRun, "avg time = ", (end-start)/NumberOfTestRun)
                print(topK, " ", nq, " ", (end-start)/NumberOfTestRun)
