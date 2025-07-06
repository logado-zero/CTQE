import faiss
import torch
import json
import os
from tqdm.auto import tqdm
from typing import List
from transformers import AutoModel

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_core.documents import Document

from bertopic import BERTopic

batch_encode_size = 256


def create_index(model_name: str="sentence-transformers/all-MiniLM-L6-v2"):
    nlp_model = HuggingFaceEmbeddings(
                                    model_name=model_name,
                                    model_kwargs={'device':'cuda:0' if torch.cuda.is_available() else 'cpu'},
                                    encode_kwargs = {'normalize_embeddings': True, 'batch_size': batch_encode_size},
                                    )

    embed = AutoModel.from_pretrained(model_name)
    len_embed = embed.config.hidden_size

    index = faiss.IndexFlatIP(len_embed)

    vector_store = FAISS(
        embedding_function=nlp_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        distance_strategy = DistanceStrategy.COSINE,
    )

    return vector_store

def load_doc(doc_path:str = "passage_corpus.json"):
    batch_doc = []
    embed_collection = None
    # Collection for Bertopic
    passage_collect = []

    with open(doc_path, "r") as f:
        ls_pid = []
        for line in tqdm(f):
            obj = json.loads(line)
            pid = obj['ref_id']

            ls_pid.append(pid)
            batch_doc.append(Document(
                page_content=obj["ref_string"],
            ))
            passage_collect.append(obj["ref_string"])

    return ls_pid,batch_doc,passage_collect

def create_bertopic_collection(passage_collect:List, model_name:str):
    topic_model = BERTopic(embedding_model=model_name)
    topics, probs = topic_model.fit_transform(passage_collect)
    
    return topic_model
def create_vector_store(doc_path:str = "passage_corpus.json",model_name: str="sentence-transformers/all-MiniLM-L6-v2"):

    save_name = model_name.split("/")[-1]
    ls_pid,batch_doc,passage_collect = load_doc(doc_path)
    vector_store = create_index(model_name)
    
    # Create vector store
    tmp_doc = []
    tmp_pid = []

    for (pid,text_embedding) in tqdm(zip(ls_pid,batch_doc),total= len(ls_pid)):
        tmp_doc.append(text_embedding)
        tmp_pid.append(pid)

        if len(tmp_pid) >= batch_encode_size:
                vector_store.add_documents(documents=tmp_doc, ids=tmp_pid)
                tmp_doc = []
                tmp_pid = []

    vector_store.add_documents(documents=tmp_doc, ids=tmp_pid)

    os.makedirs("data/vector_store_doc", exist_ok=True)
    vector_store.save_local(f"data/vector_store_doc/faiss_CORAL_{save_name}")

    # Create Bertopic collection
    topic_model = create_bertopic_collection(passage_collect=passage_collect,model_name=model_name)
    os.makedirs("data/bertopic", exist_ok=True)
    topic_model.save(f"data/bertopic/BerTopic_corpus_{save_name}")

if __name__ == '__main__':
    doc_path = "data/raw/passage_corpus.json"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    create_vector_store(doc_path=doc_path,model_name=model_name)
