############################################################################################
####################### Import libraries

import time
import os
import numpy as np
#from data_load import str2list

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


cwd = os.getcwd()
print(cwd)

print("")
import numpy as np
import pandas as pd

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, BulkInsertState
)


#np_load_old = np.load
#np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

############################################################################################
####################### Database Schema


connections.connect("default", host="localhost", port="19530")

has = utility.has_collection("test_imdb")

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
####################### Load Collections


test_imdb_hnsw_sparse = Collection("test_imdb_hnsw_sparse", schema, consistency_level="Strong")
test_imdb_hnsw_default = Collection("test_imdb_hnsw_default", schema, consistency_level="Strong")
test_imdb_hnsw_dense = Collection("test_imdb_hnsw_dense", schema, consistency_level="Strong")

#test_imdb_ivfflat = Collection("test_imdb_ivfflat", schema, consistency_level="Strong")
#test_imdb_ivfpq = Collection("test_imdb_ivfpq", schema, consistency_level="Strong")

print("Loading index...")
test_imdb_hnsw_default.load()
test_imdb_hnsw_sparse.load()
test_imdb_hnsw_dense.load()
# test_imdb_ivfflat.load()
# test_imdb_ivfpq.load()

print("Index loaded...")

############################################################################################
####################### Create Scalar Index

test_imdb_hnsw_sparse.create_index(
  field_name="avgRating", 
  index_name="scalar_index",
)

test_imdb_hnsw_dense.create_index(
  field_name="avgRating", 
  index_name="scalar_index",
)

test_imdb_hnsw_default.create_index(
  field_name="avgRating", 
  index_name="scalar_index",
)

############################################################################################
####################### Queries

def queryVector(vector, k, value=None):
    search_params = {
        "metric_type": "COSINE",
        "offset": 0,
        "ignore_growing": False,
        "params": {"ef": max(40, int(k * 2))}
    }
    
    if value is not None:
        filter_expr = f"avgRating > {value}"
    else: filter_expr = None
    
    start = time.time()
    
    res_hnsw = test_imdb_hnsw_default.search(
        data=[vector],
        anns_field="storyline_vec", 
        param=search_params,
        limit=k,
        expr=filter_expr,
        output_fields=["title_id", "primarytitle", "avgRating", "numVotes"],
        consistency_level="Strong"
    )
    
    hnsw_time = time.time() - start
    start = time.time()
    
    res_hnsw_s = test_imdb_hnsw_sparse.search(
        data=[vector],
        anns_field="storyline_vec", 
        param=search_params,
        limit=k,
        expr=filter_expr,
        output_fields=["title_id", "primarytitle", "avgRating", "numVotes"],
        consistency_level="Strong"
    )
    
    hnsw_time_s = time.time() - start
    start = time.time()
    
    res_hnsw_d = test_imdb_hnsw_dense.search(
        data=[vector],
        anns_field="storyline_vec", 
        param=search_params,
        limit=k,
        expr=filter_expr,
        output_fields=["title_id", "primarytitle", "avgRating", "numVotes"],
        consistency_level="Strong"
    )
    
    hnsw_time_d = time.time() - start
    
    #for res in res_hnsw[0]:
    #    print(type(res))
        
    return res_hnsw[0].ids, res_hnsw_s[0].ids, res_hnsw_d[0].ids, hnsw_time, hnsw_time_s, hnsw_time_d
    #return res_hnsw[0].ids, hnsw_time


#test_vec = np.load("data/storyline_vec.npy")
#print(len(test_vec[0]))
#test_vec = str2list(test_vec[:10])

#res_hnsw, res_hnsw_s, res_hnsw_d = queryVector(test_vec[0], 5, value=None)
#print("Results:\n")
#print(res_hnsw[0].ids, "\n")

#for result in res_hnsw[0]:
#    print(result.entity.get("title_id"), "\n")


############################################################################################
####################### Variables
k_vals = {1: 7.5, 2: 7.5, 5: 7.5, 10: 7.5, 20: 7.5, 50: 7.5, 100: 7.5}
#k_vals = {50: 7.5}
#filter_vals = [None, 6, 6.5, 7, 7.5, 8, 8.3, 8.52, 8.75, 9, 9.5]
#filter_vals = [8.3, 8.52, 8.75, 9, 9.5]
#filter_vals = [9.5]
columns = ["id", "prompt", "k", "selection", \
           "hnsw_runtime", "hnsw_recall", "hnsw_found",
           "hnsw_runtime_s", "hnsw_recall_s", "hnsw_found_s",
           "hnsw_runtime_d", "hnsw_recall_d", "hnsw_found_d",
          ]

############################################################################################
####################### Queries

df_top_res = pd.read_csv("data/results_df_top.csv")
df_random_res = pd.read_csv("data/results_df_random.csv")
df_prompt_res = pd.read_csv("data/results_df_prompt.csv")

#df_top_res = pd.read_csv("data/results_df_top3.csv")
#df_random_res = pd.read_csv("data/results_df_random3.csv")
#df_prompt_res = pd.read_csv("data/results_df_prompt3.csv")

top_vec = pd.read_csv("data/top_100.csv")
random_vec = pd.read_csv("data/random_100.csv")
prompt_vec = pd.read_csv("data/prompt_50.csv")

df = pd.DataFrame([], columns=columns)
count = 0

for i in range(900):
    title_id1 = df_top_res["title_id"][i]
    val1 = df_top_res["filter"][i]    
    #print("Value:", val1, "--- Type:", type(val1), "--- Check:", val1 is np.nan)
    if pd.isna(val1): val1 = None
    vec1 = top_vec[top_vec["title_id"] == title_id1]["embedding"]
    vec_list1 = vec1[i%100][1:-1].split(",")
    for n in range(len(vec_list1)):
        vec_list1[n] = float(vec_list1[n])
    
    title_id2 = df_random_res["title_id"][i]
    val2 = df_random_res["filter"][i]
    if pd.isna(val2): val2 = None
    vec2 = random_vec[random_vec["title_id"] == title_id2]["embedding"]
    #print(title_id2)
    #print(random_vec)
    #print(vec2)
    vec_list2 = vec2[i%100][1:-1].split(",")
    for n in range(len(vec_list2)):
        vec_list2[n] = float(vec_list2[n])
    
    if i < 450:
    #if i < 100:
        title_id3 = df_prompt_res["title_id"][i]
        val3 = df_prompt_res["filter"][i]
        if pd.isna(val3): val3 = None
        vec3 = prompt_vec[prompt_vec["title_id"] == title_id3]["embedding"]
        vec_list3 = vec3[i%50][1:-1].split(",")
        for n in range(len(vec_list3)):
            vec_list3[n] = float(vec_list3[n])
    
    for k in k_vals.keys():
        
        res_hnsw, res_hnsw_s, res_hnsw_d, runtime_hnsw, runtime_hnsw_s, runtime_hnsw_d = queryVector(vec_list1, k, val1)
        #res_hnsw, runtime_hnsw = queryVector(vec_list1, k, val1)
        true_res = set(df_top_res[f"k={k}"][i][2:-2].split("', '"))
        recall_hnsw = len(set(res_hnsw).intersection(true_res)) / len(true_res)
        #print(recall_hnsw)
        recall_hnsw_s = len(set(res_hnsw_s).intersection(true_res)) / len(true_res)
        """if (k == 100 or k == 20):
            print("K:", k, "-------------------------------")
            print(true_res)
            print(res_hnsw_s, "\n")"""
        recall_hnsw_d = len(set(res_hnsw_d).intersection(true_res)) / len(true_res)
        
        """
        if (i == 0 and k == 5):        
            print("---------------------------\n", "Default HNSW:", res_hnsw, recall_hnsw, "\n")
            print("Dense HNSW:", res_hnsw_d, recall_hnsw_d, "\n")
            print("Sparse HNSW:", res_hnsw_s, recall_hnsw_s, "\n")
            print(true_res, "\n", "---------------------------\n")
        """
                
        df_temp = pd.DataFrame([[title_id1, "top_100", k, val1, \
                            runtime_hnsw, recall_hnsw, len(set(res_hnsw)), \
                            runtime_hnsw_s, recall_hnsw_s, len(set(res_hnsw_s)), \
                            runtime_hnsw_d, recall_hnsw_d, len(set(res_hnsw_d))]], \
                            columns=columns
                            )
        """
        df_temp = pd.DataFrame([[title_id1, "top_100", k, val1, \
                            0, 0, 0, \
                            runtime_hnsw, recall_hnsw, 0, \
                            0, 0, 0]], \
                            columns=columns
                            )"""
        
        df = pd.concat([df, df_temp])
        
        
        res_hnsw, res_hnsw_s, res_hnsw_d, runtime_hnsw, runtime_hnsw_s, runtime_hnsw_d = queryVector(vec_list2, k, val2)
        #res_hnsw, runtime_hnsw = queryVector(vec_list, k, val2)
        true_res = set(df_random_res[f"k={k}"][i][2:-2].split("', '"))
        recall_hnsw = len(set(res_hnsw).intersection(true_res)) / len(true_res)
        recall_hnsw_s = len(set(res_hnsw_s).intersection(true_res)) / len(true_res)
        recall_hnsw_d = len(set(res_hnsw_d).intersection(true_res)) / len(true_res)
        
        df_temp = pd.DataFrame([[title_id2, "random_100", k, val2, \
                            runtime_hnsw, recall_hnsw, len(set(res_hnsw)), \
                            runtime_hnsw_s, recall_hnsw_s, len(set(res_hnsw_s)), \
                            runtime_hnsw_d, recall_hnsw_d, len(set(res_hnsw_d))]], \
                            columns=columns
                            )
        """
        df_temp = pd.DataFrame([[title_id2, "random_100", k, val2, \
                            0, 0, 0, \
                            runtime_hnsw, recall_hnsw, 0, \
                            0, 0, 0]], \
                            columns=columns
                            )
        """
        
        df = pd.concat([df, df_temp])
        
        
        if i < 450:
        
        #if i < 100:
            res_hnsw, res_hnsw_s, res_hnsw_d, runtime_hnsw, runtime_hnsw_s, runtime_hnsw_d = queryVector(vec_list3, k, val3)
            #res_hnsw, runtime_hnsw = queryVector(vec_list, k, val3)
            true_res = set(df_prompt_res[f"k={k}"][i][2:-2].split("', '"))
            recall_hnsw = len(set(res_hnsw).intersection(true_res)) / len(true_res)
        
            recall_hnsw_s = len(set(res_hnsw_s).intersection(true_res)) / len(true_res)
            recall_hnsw_d = len(set(res_hnsw_d).intersection(true_res)) / len(true_res)
        
            df_temp = pd.DataFrame([[title_id3, "prompt_100", k, val3, \
                                runtime_hnsw, recall_hnsw, len(set(res_hnsw)), \
                                runtime_hnsw_s, recall_hnsw_s, len(set(res_hnsw_s)), \
                                runtime_hnsw_d, recall_hnsw_d, len(set(res_hnsw_d))]], \
                            columns=columns
                            )
            """
            df_temp = pd.DataFrame([[title_id3, "prompt_100", k, val3, \
                                0, 0, 0, \
                                runtime_hnsw, recall_hnsw, 0, \
                                0, 0, 0]], \
                            columns=columns
                            )"""
            
            df = pd.concat([df, df_temp])
    
    count += 1
    print(count)
                
df.to_csv("imdb_milvus_tests_hnsw_idx.csv")


#K: 100
#a = {'tt8009744', 'tt0098724', 'tt0105665', 'tt6857376', 'tt0249462', 'tt0117665', 'tt1235189', 'tt0101507', 'tt6205872', 'tt0435651', 'tt0107501', 'tt0318997', 'tt0067549', 'tt4225622', 'tt1987680', 'tt0469623', 'tt0308383', 'tt2802144', 'tt0065063', 'tt1247692', 'tt0091983', 'tt0961728', 'tt0040506', 'tt1235166', 'tt0276751', 'tt1826940', 'tt0342258', 'tt0236640', 'tt0104036', 'tt0102603', 'tt2763304', 'tt0356618', 'tt0396184', 'tt4034228', 'tt1650048', 'tt0443632', 'tt0091877', 'tt0067309', 'tt1414382', 'tt0277027', 'tt6107548', 'tt0095953', 'tt0075222', 'tt14814040', 'tt2316411', 'tt2639336', 'tt0034398', 'tt0473705', 'tt0083399', 'tt0192614', 'tt0252501', 'tt0345551', 'tt1212419', 'tt3612616', 'tt0456396', 'tt9893250', 'tt0174856', 'tt0378947', 'tt0840196', 'tt8629748', 'tt0814335', 'tt4547056', 'tt3469046', 'tt0408381', 'tt0110950', 'tt0219965', 'tt1659337', 'tt0264472', 'tt2218003', 'tt0074483', 'tt1735485', 'tt0381798', 'tt10065694', 'tt0071360', 'tt0219854', 'tt0137523', 'tt0181984', 'tt0217869', 'tt7401588', 'tt3395184', 'tt0167404', 'tt2431286', 'tt2234222', 'tt1683526', 'tt0780653', 'tt0824747', 'tt0466893', 'tt5640450', 'tt1932718', 'tt0305669', 'tt0929632', 'tt9198364', 'tt0385887', 'tt5114356', 'tt0052561', 'tt0790712', 'tt13138834', 'tt3352390', 'tt1886493', 'tt0101889'}
#b = {'tt0137523', 'tt0305669', 'tt0167404', 'tt2763304', 'tt1659337', 'tt1932718', 'tt2218003', 'tt0276751', 'tt0840196', 'tt1212419', 'tt3469046', 'tt2802144', 'tt0107501', 'tt1650048', 'tt0396184', 'tt0381798', 'tt0034398', 'tt0249462', 'tt0356618', 'tt0083399', 'tt1235189', 'tt3612616', 'tt0105665', 'tt0264472', 'tt0101507', 'tt0252501', 'tt0095953', 'tt0104036', 'tt0181984', 'tt2431286', 'tt2234222', 'tt1826940', 'tt0091983', 'tt6857376', 'tt0780653', 'tt0067309', 'tt0110950', 'tt0071360', 'tt0219965', 'tt8009744', 'tt0824747', 'tt2316411', 'tt0067549', 'tt0435651', 'tt4225622', 'tt0469623', 'tt0040506', 'tt7401588', 'tt0098724', 'tt0456396', 'tt4034228', 'tt1716777', 'tt0103793', 'tt0073812', 'tt0191754', 'tt1220617', 'tt5687612', 'tt1179069', 'tt0172493', 'tt4192812', 'tt7456310', 'tt0070334', 'tt2431438', 'tt6051216', 'tt1051906', 'tt0101669', 'tt0249380', 'tt0377260', 'tt0872230', 'tt0362269', 'tt0078754', 'tt1847731', 'tt0082933', 'tt0093407', 'tt0128442', 'tt15677150', 'tt0091474', 'tt1104733', 'tt3799694', 'tt1966359', 'tt0174480', 'tt0765010', 'tt4669264', 'tt0120684', 'tt3152624', 'tt0425118', 'tt0345061', 'tt0069762', 'tt0114369', 'tt2427892', 'tt0134067', 'tt0180093', 'tt0449487', 'tt0154421', 'tt0465551', 'tt0091538', 'tt0090060', 'tt0330251', 'tt0243155', 'tt0059646'}
