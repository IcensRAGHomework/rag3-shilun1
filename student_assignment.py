import time

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"


def get_collection():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_base=gpt_emb_config['api_base'],
        api_type=gpt_emb_config['openai_type'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    if collection.count() == 0:
        csv_file = "COA_OpenData.csv"
        df = pd.read_csv(csv_file)
        required_columns = {"Name", "Type", "Address", "Tel", "City", "Town", "CreateDate", "HostWords"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV 缺少必要欄位: {required_columns - set(df.columns)}")
        for idx, row in df.iterrows():
            metadata = {
                "file_name": csv_file,
                "name": row["Name"],
                "type": row["Type"],
                "address": row["Address"],
                "tel": row["Tel"],
                "city": row["City"],
                "town": row["Town"],
                "date": int(time.mktime(time.strptime(row["CreateDate"], "%Y-%m-%d")))  # 轉換為時間戳
            }
            # 插入數據
            collection.add(
                ids=[str(idx)],
                metadatas=[metadata],
                documents=[row["HostWords"]]
            )
    return collection


def _build_filters(city=None, store_type=None, start_date=None, end_date=None):
    filters = []
    if city:
        filters.append({"city": {"$in": city}})
    if store_type:
        filters.append({"type": {"$in": store_type}})
    if start_date:
        filters.append({"date": {"$gte": start_date.timestamp()}})
    if end_date:
        filters.append({"date": {"$lte": end_date.timestamp()}})
    return None if not filters else filters[0] if len(filters) == 1 else {"$and": filters}


def run_query(collection, question, where, new_store_name=False):
    query_results = collection.query(query_texts=[question], n_results=10, where=where)
    names, distances = query_results.get("metadatas", [[]])[0], query_results.get("distances", [[]])[0]
    if new_store_name:
        return [x.get("new_store_name", x.get("name")) for x, d in sorted(zip(names, distances), key=lambda x: x[1]) if
                d < 0.2]
    else:
        return [x.get("name") for x, d in sorted(zip(names, distances), key=lambda x: x[1]) if d < 0.2]


def generate_hw01():
    collection = get_collection()
    return collection


def generate_hw02(question, city=None, store_type=None, start_date=None, end_date=None):
    collection = get_collection()
    where = _build_filters(city, store_type, start_date, end_date)
    return run_query(collection, question, where)


def generate_hw03(question, store_name, new_store_name, city=None, store_type=None):
    collection = get_collection()
    selected = collection.get(where={"name": store_name})
    metadatas = [{**item, "new_store_name": new_store_name} for item in selected.get("metadatas", [])]
    collection.upsert(ids=selected.get("ids", []), metadatas=metadatas, documents=selected.get("documents", []))
    where = _build_filters(city, store_type)
    return run_query(collection, question, where, True)


def demo(question):
    collection = get_collection()
    return collection
