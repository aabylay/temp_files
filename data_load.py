############################################################################################
####################### Choosing INDEX parameters

index_hnsw_default = {
    "index_type":"HNSW",
    "metric_type":"COSINE",
    "params":{"M":16, "efConstruction":64}
}

index_hnsw_sparse = {
    "index_type":"HNSW",
    "metric_type":"COSINE",
    "params":{"M":4, "efConstruction":16}
}

index_hnsw_dense = {
    "index_type":"HNSW",
    "metric_type":"COSINE",
    "params":{"M":100, "efConstruction":400}
}

index_ivfflat_100 = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 100},
}

index_ivfflat_10 = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 10},
}


############################################################################################
####################### Import libraries

import time
import os
cwd = os.getcwd()
print(cwd)

import numpy as np
import pandas as pd
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, BulkInsertState
)

fmt = "\n=== {:30} ===\n"


############################################################################################
####################### Database Schema

connections.connect("default", host="localhost", port="19530")

if utility.has_collection("test_imdb_hnsw_sparse"):
    utility.drop_collection("test_imdb_hnsw_sparse")
if utility.has_collection("test_imdb_hnsw_default"):
    utility.drop_collection("test_imdb_hnsw_default")
if utility.has_collection("test_imdb_hnsw_dense"):
    utility.drop_collection("test_imdb_hnsw_dense")
if utility.has_collection("test_imdb_ivfflat"):
    utility.drop_collection("test_imdb_ivfflat")
if utility.has_collection("test_imdb_ivfpq"):
    utility.drop_collection("test_imdb_ivfpq")
    

fields = [
    FieldSchema(name="title_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="primarytitle", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="avgRating", dtype=DataType.DOUBLE),
    FieldSchema(name="numVotes", dtype=DataType.INT32),
    FieldSchema(name="genres", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=50000),
    FieldSchema(name="description_vec", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="storyline", dtype=DataType.VARCHAR, max_length=50000),
    FieldSchema(name="storyline_vec", dtype=DataType.FLOAT_VECTOR, dim=384)#,
    #FieldSchema(name="synopsis", dtype=DataType.VARCHAR, max_length=50000),
    #FieldSchema(name="synopsis_vec", dtype=DataType.FLOAT_VECTOR, dim=384)#,
    #FieldSchema(name="review1", dtype=DataType.VARCHAR, max_length=50000),
    #FieldSchema(name="review1_vec", dtype=DataType.FLOAT_VECTOR, dim=384)#,
    #FieldSchema(name="review2", dtype=DataType.VARCHAR, max_length=50000),
    #FieldSchema(name="review2_vec", dtype=DataType.FLOAT_VECTOR, dim=384),
    #FieldSchema(name="review3", dtype=DataType.VARCHAR, max_length=50000),
    #FieldSchema(name="review3_vec", dtype=DataType.FLOAT_VECTOR, dim=384)
]

schema = CollectionSchema(fields, "This is a test collection of IMDb movies")

############################################################################################
####################### Collections
test_imdb_hnsw_sparse = Collection("test_imdb_hnsw_sparse", schema, consistency_level="Strong")
test_imdb_hnsw_default = Collection("test_imdb_hnsw_default", schema, consistency_level="Strong")
test_imdb_hnsw_dense = Collection("test_imdb_hnsw_dense", schema, consistency_level="Strong")

test_imdb_ivfflat_100 = Collection("test_imdb_ivfflat_100", schema, consistency_level="Strong")
test_imdb_ivfflat_10 = Collection("test_imdb_ivfflat_10", schema, consistency_level="Strong")

#############################################################################
####################### Data Loading

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def str2list(col_name): # function correctly reads vector numpy files

    for i in range(len(col_name)):
        if isinstance(col_name[i], str):
            col_name[i] = col_name[i][1:-1].split(", ")
            #col_name[i] = col_name[i]
        else:
            col_name[i] = [9] * 384
        for j in range(len(col_name[i])):
            col_name[i][j] = float(col_name[i][j])
        col_name[i] = np.array(col_name[i])
    return col_name
        
def edit_text(col_name): # function cuts long texts
    for i in range(len(col_name)):
        if isinstance(col_name[i], float):
            col_name[i] = ""
        else:
            col_name[i] = col_name[i][:min(len(col_name), 10000)]
    return col_name


description = np.load("data/description.npy").tolist()
storyline = np.load("data/storyline.npy").tolist()
synopsis = np.load("data/synopsis.npy").tolist()
review1 = np.load("data/review1.npy").tolist()
"""review2 = np.load("data/review2.npy").tolist()
 review3 = np.load("data/review3.npy").tolist()"""

description = edit_text(description)
storyline = edit_text(storyline)
synopsis = edit_text(synopsis)
review1 = edit_text(review1)
# review2 = edit_text(review2)
# review3 = edit_text(review3)


description_vec = np.load("data/description_vec.npy")
storyline_vec = np.load("data/storyline_vec.npy")
synopsis_vec = np.load("data/synopsis_vec.npy")
review1_vec = np.load("data/review1_vec.npy")
# review2 = np.load("review2.npy")
# review3 = np.load("review3.npy")

description_vec = str2list(description_vec)
storyline_vec = str2list(storyline_vec)
synopsis_vec = str2list(synopsis_vec)
review1_vec = str2list(review1_vec)
# review2_vec = str2list(review2)
# review3_vec = str2list(review3)


entities = [
    # provide the pk field because `auto_id` is set to False
    np.load("data/title_id.npy").tolist(),
    np.load("data/primarytitle.npy").tolist(),
    np.load("data/avgRating.npy").tolist(),
    np.load("data/numVotes.npy").tolist(),
    np.load("data/genres.npy").tolist(),
    description,
    description_vec,
    storyline,
    storyline_vec#,
    #synopsis,
    #synopsis_vec#,
    #review1,
    #review1_vec#,
    #review2,
    #review2_vec,
    #review3,
    #review1_vec
]


insert_result = test_imdb_hnsw_sparse.insert(entities)
test_imdb_hnsw_sparse.flush()

insert_result = test_imdb_hnsw_dense.insert(entities)
test_imdb_hnsw_dense.flush()

insert_result = test_imdb_hnsw_default.insert(entities)
test_imdb_hnsw_default.flush()

insert_result = test_imdb_ivfflat_100.insert(entities)
test_imdb_ivfflat_100.flush()

insert_result = test_imdb_ivfflat_10.insert(entities)
test_imdb_ivfflat_10.flush()

#############################################################################
####################### Create index


test_imdb_hnsw_default.create_index(
  "storyline_vec",
  index_hnsw_default
)

test_imdb_hnsw_sparse.create_index(
  "storyline_vec",
  index_hnsw_sparse
)

test_imdb_hnsw_dense.create_index(
  "storyline_vec",
  index_hnsw_dense
)


test_imdb_ivfflat_100.create_index(
  "storyline_vec",
  index_ivfflat_100
)

test_imdb_ivfflat_10.create_index(
  "storyline_vec",
  index_ivfflat_10
)

test_imdb_ivfflat_100.create_index(
  "description_vec",
  index_ivfflat_100
)

test_imdb_ivfflat_10.create_index(
  "description_vec",
  index_ivfflat_10
)


test_imdb_hnsw_default.create_index(
  "description_vec",
  index_hnsw_default
)

test_imdb_hnsw_sparse.create_index(
  "description_vec",
  index_hnsw_sparse
)

test_imdb_hnsw_dense.create_index(
  "description_vec",
  index_hnsw_dense
)


#test_imdb_hnsw_default.load()
#test_imdb_hnsw_sparse.load()
#test_imdb_hnsw_dense.load()
#test_imdb_ivfflat_100.load()
#test_imdb_ivfflat_10.load()
print("Done")