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

test_imdb_ivfflat_100 = Collection("test_imdb_ivfflat_100", schema, consistency_level="Strong")
#test_imdb_ivfflat_100.release()

print("Loading index...")
test_imdb_ivfflat_100.load()

print("Index loaded...")

############################################################################################
####################### Create Scalar Index

test_imdb_ivfflat_100.create_index(
  field_name="avgRating", 
  index_name="scalar_index",
)

print("Scalar indexes created...")
############################################################################################
####################### Queries

def queryVector(vector, k, value=None):
    if value is not None:
        filter_expr = f"avgRating > {value}"
    else: filter_expr = None
    
    search_params = {
        "metric_type": "COSINE",
        "offset": 0,
        "ignore_growing": False,
        "params": {"nprobe": 1}
    }
    
    start = time.time()
    
    res_ivf_100_1 = test_imdb_ivfflat_100.search(
        data=[vector],
        anns_field="storyline_vec", 
        param=search_params,
        limit=k,
        expr=filter_expr,
        output_fields=["title_id", "primarytitle", "avgRating", "numVotes"],
        consistency_level="Strong"
    )
    
    q_time_100_1, start = time.time() - start, time.time()
        
    ###########################################################################
        
    search_params = {
        "metric_type": "COSINE",
        "offset": 0,
        "ignore_growing": False,
        "params": {"nprobe": 10}
    }
    
    start = time.time()
    
    res_ivf_100_10 = test_imdb_ivfflat_100.search(
        data=[vector],
        anns_field="storyline_vec", 
        param=search_params,
        limit=k,
        expr=filter_expr,
        output_fields=["title_id", "primarytitle", "avgRating", "numVotes"],
        consistency_level="Strong"
    )
    
    q_time_100_10, start = time.time() - start, time.time()
        
    #for res in res_hnsw_s[0]:
    #    print(type(res))
        
    return [\
        res_ivf_100_1[0].ids, res_ivf_100_10[0].ids, \
        q_time_100_1, q_time_100_10 \
    ]
    #return res_hnsw_s[0].ids, hnsw_time_s


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
#k_vals = {1: 7.5, 2: 7.5, 5: 7.5, 10: 7.5, 20: 7.5, 50: 7.5, 100: 7.5}
k_vals = {2: 7.5}
#filter_vals = [None, 6, 6.5, 7, 7.5, 8, 8.3, 8.52, 8.75, 9, 9.5]
#filter_vals = [8.3, 8.52, 8.75, 9, 9.5]
#filter_vals = [9.5]
columns = ["id", "prompt", "k", "selection", \
           "ivf100_1_runtime", "ivf100_1_recall",
           "ivf100_10_runtime", "ivf100_10_recall"
          ]

############################################################################################
####################### Queries

#df_top_res = pd.read_csv("data/results_df_top.csv")
#df_random_res = pd.read_csv("data/results_df_random.csv")
#df_prompt_res = pd.read_csv("data/results_df_prompt.csv")

df_top_res = pd.read_csv("data/results_df_top.csv")
df_random_res = pd.read_csv("data/results_df_random.csv")
df_prompt_res = pd.read_csv("data/results_df_prompt.csv")

top_vec = pd.read_csv("data/top_100.csv")
random_vec = pd.read_csv("data/random_100.csv")
prompt_vec = pd.read_csv("data/prompt_50.csv")

df = pd.DataFrame([], columns=columns)
count = 0
#print(df_top_res.shape[0])

for i in range(df_top_res.shape[0]):

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
        
        results = queryVector(vec_list1, k, val1)        
        #res_hnsw_s, runtime_hnsw_s = queryVector(vec_list1, k, val1)
        true_res = set(df_top_res[f"k={k}"][i][2:-2].split("', '"))
        recall_ivf_100_1 = len(set(results[0]).intersection(true_res)) / len(true_res)
        recall_ivf_100_10 = len(set(results[1]).intersection(true_res)) / len(true_res)
        
        df_temp = pd.DataFrame([[title_id1, "top_100", k, val1, \
                            results[2], recall_ivf_100_1, \
                            results[3], recall_ivf_100_10]], \
                            columns=columns
                            )
        
        df = pd.concat([df, df_temp])
        
        
        results = queryVector(vec_list2, k, val2)
        #res_hnsw_s, runtime_hnsw_s = queryVector(vec_list2, k, val2)
        true_res = set(df_random_res[f"k={k}"][i][2:-2].split("', '"))
        recall_ivf_100_1 = len(set(results[0]).intersection(true_res)) / len(true_res)
        recall_ivf_100_10 = len(set(results[1]).intersection(true_res)) / len(true_res)
        
        df_temp = pd.DataFrame([[title_id2, "random_100", k, val2, \
                            results[2], recall_ivf_100_1, \
                            results[3], recall_ivf_100_10]], \
                            columns=columns
                            )

        df = pd.concat([df, df_temp])
        
        
        if i < 450:
        #if i < 100:
            results = queryVector(vec_list3, k, val3)
            #res_hnsw_s, runtime_hnsw_s = queryVector(vec_list3, k, val3)
            true_res = set(df_prompt_res[f"k={k}"][i][2:-2].split("', '"))
            recall_ivf_100_1 = len(set(results[3]).intersection(true_res)) / len(true_res)
            recall_ivf_100_10 = len(set(results[4]).intersection(true_res)) / len(true_res)
            
            df_temp = pd.DataFrame([[title_id3, "prompt_100", k, val3, \
                                results[0], recall_ivf_100_1, \
                                results[1], recall_ivf_100_10]], \
                                columns=columns
            )
            
            df = pd.concat([df, df_temp])
        
    count += 1
    print(count)
                
df.to_csv("results_df_ivf_full_idx100.csv")
print("Done")

#K: 100
#a = {'tt8009744', 'tt0098724', 'tt0105665', 'tt6857376', 'tt0249462', 'tt0117665', 'tt1235189', 'tt0101507', 'tt6205872', 'tt0435651', 'tt0107501', 'tt0318997', 'tt0067549', 'tt4225622', 'tt1987680', 'tt0469623', 'tt0308383', 'tt2802144', 'tt0065063', 'tt1247692', 'tt0091983', 'tt0961728', 'tt0040506', 'tt1235166', 'tt0276751', 'tt1826940', 'tt0342258', 'tt0236640', 'tt0104036', 'tt0102603', 'tt2763304', 'tt0356618', 'tt0396184', 'tt4034228', 'tt1650048', 'tt0443632', 'tt0091877', 'tt0067309', 'tt1414382', 'tt0277027', 'tt6107548', 'tt0095953', 'tt0075222', 'tt14814040', 'tt2316411', 'tt2639336', 'tt0034398', 'tt0473705', 'tt0083399', 'tt0192614', 'tt0252501', 'tt0345551', 'tt1212419', 'tt3612616', 'tt0456396', 'tt9893250', 'tt0174856', 'tt0378947', 'tt0840196', 'tt8629748', 'tt0814335', 'tt4547056', 'tt3469046', 'tt0408381', 'tt0110950', 'tt0219965', 'tt1659337', 'tt0264472', 'tt2218003', 'tt0074483', 'tt1735485', 'tt0381798', 'tt10065694', 'tt0071360', 'tt0219854', 'tt0137523', 'tt0181984', 'tt0217869', 'tt7401588', 'tt3395184', 'tt0167404', 'tt2431286', 'tt2234222', 'tt1683526', 'tt0780653', 'tt0824747', 'tt0466893', 'tt5640450', 'tt1932718', 'tt0305669', 'tt0929632', 'tt9198364', 'tt0385887', 'tt5114356', 'tt0052561', 'tt0790712', 'tt13138834', 'tt3352390', 'tt1886493', 'tt0101889'}
#b = {'tt0137523', 'tt0305669', 'tt0167404', 'tt2763304', 'tt1659337', 'tt1932718', 'tt2218003', 'tt0276751', 'tt0840196', 'tt1212419', 'tt3469046', 'tt2802144', 'tt0107501', 'tt1650048', 'tt0396184', 'tt0381798', 'tt0034398', 'tt0249462', 'tt0356618', 'tt0083399', 'tt1235189', 'tt3612616', 'tt0105665', 'tt0264472', 'tt0101507', 'tt0252501', 'tt0095953', 'tt0104036', 'tt0181984', 'tt2431286', 'tt2234222', 'tt1826940', 'tt0091983', 'tt6857376', 'tt0780653', 'tt0067309', 'tt0110950', 'tt0071360', 'tt0219965', 'tt8009744', 'tt0824747', 'tt2316411', 'tt0067549', 'tt0435651', 'tt4225622', 'tt0469623', 'tt0040506', 'tt7401588', 'tt0098724', 'tt0456396', 'tt4034228', 'tt1716777', 'tt0103793', 'tt0073812', 'tt0191754', 'tt1220617', 'tt5687612', 'tt1179069', 'tt0172493', 'tt4192812', 'tt7456310', 'tt0070334', 'tt2431438', 'tt6051216', 'tt1051906', 'tt0101669', 'tt0249380', 'tt0377260', 'tt0872230', 'tt0362269', 'tt0078754', 'tt1847731', 'tt0082933', 'tt0093407', 'tt0128442', 'tt15677150', 'tt0091474', 'tt1104733', 'tt3799694', 'tt1966359', 'tt0174480', 'tt0765010', 'tt4669264', 'tt0120684', 'tt3152624', 'tt0425118', 'tt0345061', 'tt0069762', 'tt0114369', 'tt2427892', 'tt0134067', 'tt0180093', 'tt0449487', 'tt0154421', 'tt0465551', 'tt0091538', 'tt0090060', 'tt0330251', 'tt0243155', 'tt0059646'}
