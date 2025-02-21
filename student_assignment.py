import datetime
import chromadb
import traceback
import pandas as pd
import time


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
    return collection


def generate_hw01():
    collection = get_collection()
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


def generate_hw02(question, city, store_type, start_date, end_date):
    pass


def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass


def demo(question):
    collection = get_collection()
    return collection
