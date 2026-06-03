import argparse
import json
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import pytrec_eval
import Stemmer
import torch
import torch.nn.functional as F
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.retrievers.bm25 import BM25Retriever
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from model.ctqe import CTQE


SEED = 1212
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HISTORY_NUM = 2
DEFAULT_BATCH_SIZE = 64
DEFAULT_EVAL_TOP_K = 20
DEFAULT_BM25_TOP_K = 25
DEFAULT_OUTPUT_DIR = Path("result") / "ctqe"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_paths(base_dir: Path, model_name: str) -> dict:
    model_short = model_name.split("/")[-1]

    data_root = base_dir / "data"
    dataset_dir = first_existing_path(data_root / "raw" / "dataset", data_root / "raw")

    return {
        "data_root": data_root,
        "dataset_dir": dataset_dir,
        "train_file": first_existing_path(
            dataset_dir / "train" / "train_conversation.json",
            dataset_dir / "train" / "new_train_conversation.json",
        ),
        "test_file": first_existing_path(
            dataset_dir / "test" / "test_conversation.json",
            dataset_dir / "test" / "new_test_conversation.json",
        ),
        "passage_file": first_existing_path(
            dataset_dir / "passage_corpus.json",
            data_root / "raw" / "passage_corpus.json",
        ),
        "vector_store_path": data_root / "vector_store_doc" / f"faiss_CORAL_{model_short}",
        "processed_dir": data_root / "processed",
        "output_dir": base_dir / DEFAULT_OUTPUT_DIR,
        "checkpoint_path": base_dir / "result" / "ctqe" / "ctqe_most.pt",
    }


def load_conversations(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def create_bm25_retriever(collection_path: Path, top_k: int = DEFAULT_BM25_TOP_K) -> BM25Retriever:
    docs = []
    with open(collection_path, "r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="BM25 corpus"):
            passage = json.loads(line)
            docs.append(LlamaDocument(metadata={"id_": passage["ref_id"]}, text=passage["ref_string"]))

    parser = SimpleFileNodeParser()
    nodes = parser.get_nodes_from_documents(docs)

    return BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )


def create_vector_store(doc_path: Path, model_name: str, vector_store_path: Path) -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda:0" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 256},
    )
    embed_model = AutoModel.from_pretrained(model_name)
    dim = embed_model.config.hidden_size

    index = faiss.IndexFlatIP(dim)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        distance_strategy=DistanceStrategy.COSINE,
    )

    batch_docs = []
    batch_ids = []
    with open(doc_path, "r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="FAISS docs"):
            passage = json.loads(line)
            batch_docs.append(Document(page_content=passage["ref_string"]))
            batch_ids.append(str(passage["ref_id"]))
            if len(batch_docs) >= 256:
                vector_store.add_documents(documents=batch_docs, ids=batch_ids)
                batch_docs = []
                batch_ids = []

    if batch_docs:
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)

    vector_store_path.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(vector_store_path))
    return vector_store


def load_or_build_vector_store(doc_path: Path, model_name: str, vector_store_path: Path, build_if_missing: bool) -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda:0" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 256},
    )

    if vector_store_path.exists():
        return FAISS.load_local(
            str(vector_store_path),
            embeddings,
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.COSINE,
        )

    if not build_if_missing:
        raise FileNotFoundError(f"Missing FAISS index at {vector_store_path}")

    return create_vector_store(doc_path, model_name, vector_store_path)


def build_records(conversations: list, history_num: int, bm25_retriever: BM25Retriever) -> list:
    records = []
    for conversation in tqdm(conversations, desc="Preprocess conversations"):
        conv_id = conversation["conv_id"]
        turns = conversation["turns"]
        history_session = []

        for turn in turns:
            nodes = bm25_retriever.retrieve(turn["question"]) if bm25_retriever is not None else []
            irrelevant_docs_pids = [
                str(node.metadata["id_"])
                for node in nodes
                if str(node.metadata["id_"]) not in turn.get("golden_docs_pids", [])
            ][:10]

            history_slice = history_session[-history_num:]
            context_session = "[SEP]" + "[SEP]".join(history_slice)

            records.append(
                {
                    "id": f"{conv_id}_{turn['turn_id']}",
                    "question": turn["question"],
                    "rewrite": turn.get("golden_rewrite"),
                    "relevant_docs_pids": turn.get("golden_docs_pids", []),
                    "irrelevant_docs_pids": irrelevant_docs_pids,
                    "context_session": context_session,
                }
            )

            history_session.append(turn["question"])

    return records


class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, records: list):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def get_collate_fn(vector_store, tokenizer, model_embedding):
    index_to_docstore_id = vector_store.index_to_docstore_id
    docstore_id_to_index = {docstore_id: index for index, docstore_id in index_to_docstore_id.items()}
    dim = vector_store.index.d

    def get_vectors(ids):
        if len(ids) == 0:
            return torch.zeros(1, dim)
        vectors = []
        for passage_id in ids:
            index = docstore_id_to_index[str(passage_id)]
            vectors.append(torch.tensor(vector_store.index.reconstruct_n(index, 1)[0]))
        return torch.stack(vectors, dim=0)

    def collate_fn(batch):
        question_id = [item["id"] for item in batch]
        question_text = [item["question"] for item in batch]
        query_token_ids = tokenizer(question_text, padding=True, truncation=True, return_tensors="pt").to(model_embedding.device)

        relevant_docs_pids = [item["relevant_docs_pids"] for item in batch]
        irrelevant_docs_pids = [item["irrelevant_docs_pids"] for item in batch]
        batch_relevant = [get_vectors(ids).to(model_embedding.device) if len(ids) != 0 else None for ids in relevant_docs_pids]
        batch_irrelevant = torch.stack([get_vectors(ids) for ids in irrelevant_docs_pids], dim=0).to(model_embedding.device)

        session = [item["context_session"] for item in batch]
        session_token_ids = tokenizer(session, padding=True, truncation=True, return_tensors="pt").to(model_embedding.device)

        with torch.no_grad():
            embed_query = model_embedding(**query_token_ids)
            embed_sess = model_embedding(**session_token_ids)

        mean_query = mean_pooling(embed_query, query_token_ids["attention_mask"])
        mean_context = mean_pooling(embed_sess, session_token_ids["attention_mask"]).unsqueeze(1)
        context = embed_sess[0]

        mean_rewrite = None
        if batch[0]["rewrite"] is not None:
            rewrite = [item["rewrite"] for item in batch]
            rewrite_token_ids = tokenizer(rewrite, padding=True, truncation=True, return_tensors="pt").to(model_embedding.device)
            with torch.no_grad():
                embed_rewrite = model_embedding(**rewrite_token_ids)
            mean_rewrite = mean_pooling(embed_rewrite, rewrite_token_ids["attention_mask"])

        return {
            "question_id": question_id,
            "question_text": question_text,
            "embed_query": embed_query[0],
            "mean_query": mean_query,
            "context": context,
            "mean_context": mean_context.squeeze(1),
            "rewrite": mean_rewrite,
            "relevant_docs_pids": relevant_docs_pids,
            "batch_relevant": batch_relevant,
            "batch_irrelevant": batch_irrelevant,
        }

    return collate_fn


def evaluate_run(res_model, res_gold):
    metrics = [
        "recall_5",
        "recall_10",
        "recall_20",
        "ndcg_cut_3",
        "map",
        "map_cut_10",
        "recip_rank",
    ]
    evaluator = pytrec_eval.RelevanceEvaluator(res_gold, set(metrics))
    out = {metric: [] for metric in metrics}
    for _, scores in evaluator.evaluate(res_model).items():
        for metric in metrics:
            out[metric].append(scores[metric])
    return {metric: float(np.mean(out[metric])) for metric in metrics}


def evaluate_by_vector(vector_store, embeddings, batch_ids, batch_gold, top_k):
    res_model = {}
    res_gold = {}
    for i, qid in enumerate(batch_ids):
        gold = {str(pid): 1 for pid in batch_gold[i]}
        nodes = vector_store.similarity_search_with_score_by_vector(embedding=embeddings[i].tolist(), k=top_k)
        res_model[qid] = {node[0].id: float(node[1]) for node in nodes}
        res_gold[qid] = gold
    return res_model, res_gold


def evaluate_ctqe(model, vector_store, loader, top_k):
    model.eval()
    res_model = {}
    res_gold = {}
    time_inference = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            mean_ids = batch["mean_query"].to(model.device)
            ids = batch["embed_query"].to(model.device)
            mean_ctx = batch["mean_context"].to(model.device)
            ctx = batch["context"].to(model.device)
            question_id = batch["question_id"]
            relevant_docs_pids = batch["relevant_docs_pids"]

            start = time.time()
            output = model(mean_ids, ids, mean_ctx, ctx)
            embed_question = F.normalize(output, p=2, dim=1)
            time_inference += time.time() - start
            count += len(question_id)

            for i, qid in enumerate(question_id):
                gold = {str(pid): 1 for pid in relevant_docs_pids[i]}
                nodes = vector_store.similarity_search_with_score_by_vector(
                    embedding=embed_question[i].tolist(),
                    k=top_k,
                )
                res_gold[qid] = gold
                res_model[qid] = {node[0].id: float(node[1]) for node in nodes}

    results = evaluate_run(res_model, res_gold)
    results["time_inference_total"] = time_inference
    results["time_inference_per_query"] = time_inference / max(count, 1)
    return results


def evaluate_dense_baseline(vector_store, loader, top_k):
    res_model = {}
    res_gold = {}
    for batch in loader:
        embeddings = F.normalize(batch["mean_query"], p=2, dim=1)
        question_id = batch["question_id"]
        relevant_docs_pids = batch["relevant_docs_pids"]
        batch_res_model, batch_res_gold = evaluate_by_vector(
            vector_store, embeddings, question_id, relevant_docs_pids, top_k
        )
        res_model.update(batch_res_model)
        res_gold.update(batch_res_gold)
    return evaluate_run(res_model, res_gold)


def evaluate_rewrite_baseline(vector_store, loader, top_k):
    res_model = {}
    res_gold = {}
    for batch in loader:
        embeddings = batch["rewrite"]
        if embeddings is None:
            continue
        embeddings = F.normalize(embeddings, p=2, dim=1)
        question_id = batch["question_id"]
        relevant_docs_pids = batch["relevant_docs_pids"]
        batch_res_model, batch_res_gold = evaluate_by_vector(
            vector_store, embeddings, question_id, relevant_docs_pids, top_k
        )
        res_model.update(batch_res_model)
        res_gold.update(batch_res_gold)
    if not res_model:
        return {}
    return evaluate_run(res_model, res_gold)


def evaluate_bm25(loader, bm25_retriever, top_k):
    res_model = {}
    res_gold = {}
    for batch in loader:
        question_id = batch["question_id"]
        question_text = batch["question_text"]
        relevant_docs_pids = batch["relevant_docs_pids"]
        for i, qid in enumerate(question_id):
            nodes = bm25_retriever.retrieve(question_text[i])
            res_model[qid] = {str(node.metadata["id_"]): float(node.score) for node in nodes[:top_k]}
            res_gold[qid] = {str(pid): 1 for pid in relevant_docs_pids[i]}
    return evaluate_run(res_model, res_gold)


def build_model(model_name: str, checkpoint_path: Path, device: torch.device, num_heads: int = 8, dropout_rate: float = 0.25):
    embedding_model = AutoModel.from_pretrained(model_name).to(device)
    lstm_param = {"n_layers": 2, "bidirectional": True}
    model = CTQE(embedding_model.config.hidden_size, dropout_rate, lstm_param, num_heads)
    model.to(device)
    model.device = device

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint at {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model, embedding_model


def cache_records(records: list, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as handle:
        pickle.dump(records, handle)


def load_records(cache_path: Path):
    with open(cache_path, "rb") as handle:
        return pickle.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CTQE and retrieval baselines.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--history-num", type=int, default=DEFAULT_HISTORY_NUM)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_EVAL_TOP_K)
    parser.add_argument("--bm25-top-k", type=int, default=DEFAULT_BM25_TOP_K)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--build-faiss", action="store_true", help="Build the FAISS index if it does not exist.")
    parser.add_argument("--no-build-faiss", action="store_true", help="Fail if the FAISS index is missing.")
    parser.add_argument("--cache-records", action="store_true", help="Cache preprocessed evaluation records.")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path.cwd()
    paths = resolve_paths(base_dir, args.model_name)

    output_dir = Path(args.output_dir) if args.output_dir else paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else paths["checkpoint_path"]
    vector_store = load_or_build_vector_store(
        paths["passage_file"],
        args.model_name,
        paths["vector_store_path"],
        build_if_missing=args.build_faiss and not args.no_build_faiss,
    )

    bm25_retriever = create_bm25_retriever(paths["passage_file"], top_k=args.bm25_top_k)

    cache_path = paths["processed_dir"] / "evaluation_records.pkl"
    if args.cache_records and cache_path.exists():
        records = load_records(cache_path)
    else:
        conversations = load_conversations(paths["test_file"])
        records = build_records(conversations, args.history_num, bm25_retriever)
        if args.cache_records:
            cache_records(records, cache_path)

    dataset = ConversationDataset(records)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model, embedding_model = build_model(args.model_name, checkpoint_path, device)
    collate_fn = get_collate_fn(vector_store, tokenizer, embedding_model)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    results = {
        "ctqe": evaluate_ctqe(model, vector_store, loader, args.top_k),
        "dense_baseline": evaluate_dense_baseline(vector_store, loader, args.top_k),
        "bm25_baseline": evaluate_bm25(loader, bm25_retriever, args.top_k),
    }

    if any(record["rewrite"] is not None for record in records):
        results["rewrite_baseline"] = evaluate_rewrite_baseline(vector_store, loader, args.top_k)

    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()