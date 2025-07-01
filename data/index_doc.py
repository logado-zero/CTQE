import faiss
import torch
import json
import os
from tqdm.auto import tqdm
from transformers import AutoModel

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import DistanceStrategy

from langchain_core.documents import Document

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

    with open(doc_path, "r") as f:
        ls_pid = []
        for line in tqdm(f):
            obj = json.loads(line)
            pid = obj['ref_id']

            ls_pid.append(pid)
            batch_doc.append(Document(
                page_content=obj["ref_string"],
            ))

    return ls_pid,batch_doc

def create_vector_store(doc_path:str = "passage_corpus.json",model_name: str="sentence-transformers/all-MiniLM-L6-v2"):

    ls_pid,batch_doc = load_doc(doc_path)
    vector_store = create_index(model_name)
    
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
    vector_store.save_local("data/vector_store_doc/faiss_CORAL_MiniLML6_v2")

if __name__ == '__main__':
    doc_path = "data/raw/passage_corpus.json"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    create_vector_store(doc_path=doc_path,model_name=model_name)
