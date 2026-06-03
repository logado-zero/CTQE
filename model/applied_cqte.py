from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:
    from bertopic import BERTopic
    from bertopic.backend import BaseEmbedder
except Exception:  # pragma: no cover - optional dependency
    BERTopic = None
    BaseEmbedder = object

from model.ctqe import CTQE as CTQEModel


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHECKPOINT_PATH = Path("result") / "ctqe" / "ctqe_most.pt"
DEFAULT_BERTOPIC_DIR = Path("result") / "ctqe"
DEFAULT_HISTORY_NUM = 2
DEFAULT_DROPOUT_RATE = 0.25
DEFAULT_NUM_HEADS = 8
DEFAULT_LSTM_LAYERS = 2


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def _load_corpus_texts(corpus_path: Path) -> list[str]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus file at {corpus_path}")

    try:
        with open(corpus_path, "r", encoding="utf-8") as handle:
            corpus = json.load(handle)
    except json.JSONDecodeError:
        corpus = []
        with open(corpus_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                corpus.append(json.loads(line))

    texts: list[str] = []
    for item in corpus:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            texts.append(item.get("ref_string") or item.get("text") or item.get("content") or "")
        else:
            texts.append(str(item))
    return [text for text in texts if text]


class _HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, tokenizer, model, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def embed(self, documents, verbose: bool = False):
        token_ids = self.tokenizer(
            list(documents),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            output = self.model(**token_ids)
        embeddings = mean_pooling(output, token_ids["attention_mask"])
        return embeddings.detach().cpu().numpy()


class CTQEConversationApp:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        corpus_path: str | Path | None = None,
        use_bertopic: bool = False,
        bertopic_path: str | Path | None = None,
        history_num: int = DEFAULT_HISTORY_NUM,
        dropout_rate: float = DEFAULT_DROPOUT_RATE,
        num_heads: int = DEFAULT_NUM_HEADS,
        lstm_layers: int = DEFAULT_LSTM_LAYERS,
        bidirectional: bool = True,
        device: str | torch.device | None = None,
    ):
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path)
        self.corpus_path = Path(corpus_path) if corpus_path is not None else None
        self.use_bertopic = use_bertopic
        self.bertopic_path = Path(bertopic_path) if bertopic_path is not None else None
        self.history_num = history_num
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.embedding_model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.embedding_model.eval()
        self.bertopic_backend = _HuggingFaceEmbedder(self.tokenizer, self.embedding_model, self.device)

        lstm_param = {"n_layers": lstm_layers, "bidirectional": bidirectional}
        self.model = CTQEModel(
            self.embedding_model.config.hidden_size,
            dropout_rate,
            lstm_param,
            num_heads=num_heads,
        ).to(self.device)
        self.model.device = self.device
        self._load_checkpoint()
        self.model.eval()

        self.bertopic = None
        self.corpus_texts: list[str] = []
        if self.use_bertopic:
            self.bertopic = self._load_or_train_bertopic()

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint at {self.checkpoint_path}")

        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def _default_bertopic_path(self) -> Path:
        model_short = self.model_name.split("/")[-1]
        return DEFAULT_BERTOPIC_DIR / f"BerTopic_corpus_{model_short}"

    def _load_or_train_bertopic(self):
        if BERTopic is None:
            raise RuntimeError("BERTopic is not available. Install bertopic or disable use_bertopic.")

        bertopic_path = self.bertopic_path or self._default_bertopic_path()
        if bertopic_path.exists():
            topic_model = BERTopic.load(str(bertopic_path))
            topic_model.embedding_model = self.bertopic_backend
            return topic_model

        if self.corpus_path is None:
            raise ValueError("corpus_path is required when use_bertopic is enabled and no checkpoint exists")

        self.corpus_texts = _load_corpus_texts(self.corpus_path)
        if not self.corpus_texts:
            raise ValueError(f"No usable texts were found in corpus_path={self.corpus_path}")

        topic_model = BERTopic(embedding_model=self.bertopic_backend)
        topic_model.fit_transform(self.corpus_texts)
        bertopic_path.parent.mkdir(parents=True, exist_ok=True)
        topic_model.save(str(bertopic_path))
        self.bertopic_path = bertopic_path
        topic_model.embedding_model = self.bertopic_backend
        return topic_model

    def _tokenize_texts(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    def _encode_texts(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = self._tokenize_texts(texts)
        with torch.inference_mode():
            output = self.embedding_model(**token_ids)
        return output[0], mean_pooling(output, token_ids["attention_mask"])

    def _topic_embedding(self, topic_id: int) -> torch.Tensor:
        if self.bertopic is None:
            raise RuntimeError("BERTopic model is not initialized")

        topic_embeddings = getattr(self.bertopic, "topic_embeddings_", None)
        if topic_embeddings is None:
            raise RuntimeError("BERTopic topic embeddings are not available on the loaded model")

        if topic_id is None or topic_id < 0 or topic_id >= len(topic_embeddings):
            return torch.zeros(self.embedding_model.config.hidden_size, device=self.device)
        return torch.as_tensor(topic_embeddings[topic_id], device=self.device, dtype=torch.float32)

    def _build_context(self, context_history: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        history_full: list[str] = []
        history_questions: list[str] = []

        if context_history is not None:
            for item in context_history:
                if item is None:
                    continue
                q, r = None, None
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    q, r = item[0], item[1]
                elif isinstance(item, dict) and "question" in item and "response" in item:
                    q = item["question"]
                    r = item["response"]
                elif isinstance(item, str):
                    if "[SEP]" in item:
                        parts = item.split("[SEP]")
                        q = parts[0].strip()
                        r = parts[1].strip() if len(parts) > 1 else ""
                    else:
                        q = item
                        r = ""

                if q:
                    history_questions.append(q)
                    history_full.append(f"{q} [SEP] {r}" if r else q)

        if not history_questions:
            history_questions = [""]
            history_full = [""]

        session_text = "[SEP]".join(history_questions)
        ctx_tokens, ctx_mean = self._encode_texts([session_text])

        if not self.use_bertopic or self.bertopic is None:
            return ctx_tokens, ctx_mean

        topics, _ = self.bertopic.transform(history_full)
        topic_vectors = torch.stack([self._topic_embedding(int(topic_id)) for topic_id in topics], dim=0)
        context = torch.cat([topic_vectors.unsqueeze(0), ctx_tokens], dim=1)
        mean_context = topic_vectors.mean(dim=0, keepdim=True)
        return context, mean_context

    def run(
        self,
        query: str,
        context_history: Iterable[str] | None = None,
        normalize_output: bool = False,
    ) -> torch.Tensor:
        history = list(context_history or [])[-self.history_num :]
        query_tokens = self._tokenize_texts([query])
        with torch.inference_mode():
            query_output = self.embedding_model(**query_tokens)

        mean_query = mean_pooling(query_output, query_tokens["attention_mask"])
        query_ctx = query_output[0]
        context, mean_context = self._build_context(history)

        with torch.inference_mode():
            output = self.model(mean_query, query_ctx, mean_context, context)

        if normalize_output:
            output = F.normalize(output, p=2, dim=1)
        return output.squeeze(0)

    def __call__(
        self,
        query: str,
        context_history: Iterable[str] | None = None,
        normalize_output: bool = False,
    ) -> torch.Tensor:
        return self.run(query, context_history=context_history, normalize_output=normalize_output)



if __name__ == "__main__":
    bertopic_path = Path(r"C:\Users\admin\Documents\Code\WorkSpace\CTQE\data\bertopic\BerTopic_corpus_mpnet_v2no_emb")
    corpus_path = Path(r"C:\Users\admin\Documents\Code\WorkSpace\CTQE\data\raw\dataset\passage_corpus.json")
    checkpoint_path = Path(r"C:\Users\admin\Documents\Code\WorkSpace\CTQE\model\checkpoints\merge_emb_mpnet_complete2.pt")

    app = CTQEConversationApp(
        model_name="sentence-transformers/all-mpnet-base-v2",
        checkpoint_path=checkpoint_path,
        corpus_path=corpus_path,
        use_bertopic=True,
        bertopic_path=bertopic_path,
        history_num=2,
    )

    # History entries are (question, response) pairs
    history = [
        ("Where can I find details about the mpnet-based embeddings?", "Check the model card and the sentence-transformers docs."),
        ("Is there an example combining topics and embeddings?", "Yes — you can precompute topic embeddings with BERTopic and concatenate them."),
    ]
    current_question = "How can I combine conversation history with the current query for CTQE inference?"

    query_tokens = app._tokenize_texts([current_question])
    with torch.inference_mode():
        query_output = app.embedding_model(**query_tokens)
    query_embedding = mean_pooling(query_output, query_tokens["attention_mask"])

    transformed_question = app.run(current_question, history, normalize_output=False).unsqueeze(0)
    cosine_score = F.cosine_similarity(query_embedding, transformed_question, dim=1).item()

    print("History:")
    for item in history:
        print(f"- {item}")
    print(f"Current question: {current_question}")
    print(f"Transformed embedding shape: {tuple(transformed_question.shape)}")
    print(f"Cosine score between query and transformed question: {cosine_score:.6f}")



