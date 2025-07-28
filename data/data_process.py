import json
from typing import List, Dict
from tqdm.auto import tqdm
import pickle
import os

import torch

import faiss
import Stemmer

from langchain_core.documents import Document
from llama_index.core import Document as Doc_llama_index
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore

from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.retrievers.bm25 import BM25Retriever

from bertopic import BERTopic
from transformers import AutoTokenizer, AutoModel

class PreprocessedDataset():
  """
  Preprocess the dataset for training and evaluation.
  It reads the conversation data, retrieves relevant documents using BM25,
  """
  def __init__(self, filename, history_num, training=True, vs_context_path = None, bertopic_context_path=None, \
               model_embedding_name: str ="sentence-transformers/all-MiniLM-L6-v2", batch_encode_size: int = 128,\
                collection_path:str = "passage_corpus.json"):
    """
    Initialize the PreprocessedDataset with the given parameters.
    Args:
        filename (str): Path to the JSON file containing conversation data.
        history_num (int): Number of previous turns to consider as context.
        training (bool): Whether the dataset is for training or evaluation.
        vs_context_path (str): Path to the vector store context, if available.
        bertopic_context_path (str): Path to the Bertopic context, if available.
        model_embedding_name (str): Name of the model to use for embeddings.
        batch_encode_size (int): Size of batches for encoding documents.
        collection_path (str): Path to the collection of passages for BM25 retrieval.
    """
    self._filename = filename
    self._history_num = history_num
    self.training = training
    self.vs_context_path = vs_context_path
    self.bertopic_context_path = bertopic_context_path
    self.batch_encode_size = batch_encode_size
    self.collection_path = collection_path
    self.model_embedding_name = model_embedding_name
    # Initialize the embedding model
    self.model_embedding = HuggingFaceEmbeddings(
                                model_name=self.model_embedding_name,
                                model_kwargs={'device':'cuda:0' if torch.cuda.is_available() else 'cpu'},
                                encode_kwargs = {'normalize_embeddings': True, 'batch_size': batch_encode_size}
                                )
    # Initialize the FAISS index
    self.index_ctx = faiss.IndexFlatIP(self.model_embedding._client.get_sentence_embedding_dimension())
    # Load the conversation data from the specified file
    with open(filename, "r") as f:
      self._total_data = json.load(f)

    # Initialize the vector store context
    self.vs_context = FAISS(
          embedding_function=self.model_embedding,
          index=self.index_ctx,
          docstore=InMemoryDocstore(),
          index_to_docstore_id={},
          distance_strategy = DistanceStrategy.COSINE,
      ) if self.vs_context_path is None else FAISS.load_local( self.vs_context_path, self.model_embedding, allow_dangerous_deserialization=True, distance_strategy = DistanceStrategy.COSINE)
    
    self.bertopic = None
    if self.bertopic_context_path is not None:
      self.bertopic = BERTopic.load(self.bertopic_context_path)


  def batch(self,iterable, n=1):
    """
    Yield successive n-sized chunks from iterable.
    Args:
        iterable (iterable): The input iterable to be chunked.
        n (int): The size of each chunk.
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

  def __call__(self) -> List[Dict]:
    """
    Process the dataset to create a list of dictionaries containing conversation turns and their contexts.
    Returns:
        List[Dict]: A list of dictionaries where each dictionary represents a conversation turn with its context.
    """
    res_ls = []
    res_ls_only_2 = []
    context_id = 0
    ctx_ids = []
    context_ls = []
    context_ls.append(Document(page_content=""))
    context_id += 1

    #Create the BM25 from collection
    bm25_retriever = create_BM25_retriever(collection_path=self.collection_path)

    for cnv in tqdm(self._total_data,desc=f"Preprocessing Dataset from file {self._filename} ...."):
      cnv_name = cnv['conv_id']
      turns = cnv['turns']
      history_ls = []
      history_ctx_ids = []
      history_session = []
      cnt_turn = 0
      for turn in turns:
          context = history_ls[-self._history_num:]
          ctx_ids = history_ctx_ids[-self._history_num:]
          ctx_sess = history_session[-self._history_num:]


          if len(context) <  self._history_num:
            times = (self._history_num-len(context))
            context.extend([""]*times)
            ctx_ids.extend([0]*times)

          
          nodes = bm25_retriever.retrieve(turn['question'])
          irrelevant_docs_pids = list(filter(lambda v: v not in turn['golden_docs_pids'], [str(node.metadata["id_"]) for node in nodes]))[:10]

          if self.bertopic_context_path:
            topics, _ = self.bertopic.transform(context)
          else: topics = None
          result_turn ={
                  "id":       cnv_name + "_" + str(turn['turn_id']),
                  "question": turn['question'],
                  "rewrite":  turn['golden_rewrite'] if self.training else None,
                  "relevant_docs_pids": turn['golden_docs_pids'] if len(turn['golden_docs_pids']) is not None else None,
                  "irrelevant_docs_pids": irrelevant_docs_pids,
                  "context": context,
                  "context_ids": ctx_ids,
                  "context_topics": topics,
                  "context_session": "[SEP]" + "[SEP]".join(ctx_sess),
              }


          res_ls.append(result_turn)
          if cnt_turn < 3:
            cnt_turn += 1
            res_ls_only_2.append(result_turn)

          history_ls.append(turn['question'] + " [SEP] " + turn['response'])
          history_ctx_ids.append(context_id)
          history_session.append(turn['question'])
          context_id += 1
          context_ls.append(Document(page_content=history_ls[-1]))

    times = 0
    if  self.vs_context_path is None and self.bertopic_context_path is None:
      for docs in tqdm(self.batch(context_ls,self.batch_encode_size*2), desc="Add documents...", total=len(context_ls)//(self.batch_encode_size*2)):
        num_docs = len(docs)
        if num_docs == self.batch_encode_size*2:
          self.vs_context.add_documents(documents=docs, ids=list(range(times*num_docs,(times+1)*num_docs)))
        else: self.vs_context.add_documents(documents=docs, ids=list(range(times*self.batch_encode_size*2,times*self.batch_encode_size*2+num_docs)))
        times +=1

    return res_ls, res_ls_only_2
  

def create_BM25_retriever(collection_path:str = "passage_corpus.json"):
  """
  Create a BM25 retriever from the provided collection path.
  Args: 
      collection_path (str): Path to the JSON file containing the collection of passages.
  Returns:
      BM25Retriever: A BM25 retriever initialized with the passages from the collection.
  """
  docs = []
  with open(collection_path) as handle:
      for line in tqdm(handle):
          passage = json.loads(line)
          docs.append(Doc_llama_index(metadata={"id_":passage["ref_id"]}, text=passage["ref_string"]))

  parser = SimpleFileNodeParser()
  md_nodes = parser.get_nodes_from_documents(docs)

  bm25_retriever = BM25Retriever.from_defaults(
      nodes=md_nodes,
      similarity_top_k=25,
      # Optional: We can pass in the stemmer and set the language for stopwords
      # This is important for removing stopwords and stemming the query + text
      # The default is english for both
      stemmer=Stemmer.Stemmer("english"),
      language="english",
  )

  return bm25_retriever


class RetrieverDataset(torch.utils.data.Dataset):
    """
    Dataset class for retrieving and processing conversation data for training, evaluation.
    It handles the loading of conversation data, retrieval of relevant documents, and preparation of features.
    """
    def __init__(self, filename,history_num=2, training=True, vs_context_path = None, bertopic_context_path=None,only_2 = False, pre_data_path = None,
                 model_embedding_name: str ="sentence-transformers/all-MiniLM-L6-v2", collection_path:str = "passage_corpus.json"):
        """
        Initialize the RetrieverDataset with the given parameters.
        Args: 
            filename (str): Path to the JSON file containing conversation data.
            history_num (int): Number of previous turns to consider as context.
            training (bool): Whether the dataset is for training or evaluation.
            vs_context_path (str): Path to the vector store context, if available.
            bertopic_context_path (str): Path to the Bertopic context, if available.
            only_2 (bool): If True, only use the first two turns of each conversation.
            pre_data_path (str): Path to preprocessed data, if available.
            model_embedding_name (str): Name of the model to use for embeddings.
            collection_path (str): Path to the collection of passages for BM25 retrieval.
        """

        self._filename = filename
        self._history_num = history_num
        self.training = training
        self.vs_context_path = vs_context_path
        self.bertopic_context_path = bertopic_context_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_embedding_name)
        self.embedding_model = AutoModel.from_pretrained(model_embedding_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model.to(device)

        # Initialize the dataset
        self.total_data = PreprocessedDataset(self._filename,self._history_num,self.training, self.vs_context_path, self.bertopic_context_path,
                                              model_embedding_name="sentence-transformers/all-MiniLM-L6-v2", collection_path=collection_path)
        if pre_data_path is None:
          self.pre_data, self.pre_data_only_2 = self.total_data()
        else:
          with open(pre_data_path, 'rb') as handle:
            self.pre_data = pickle.load(handle)
          self.pre_data_only_2 = None

        self.only_2 = only_2

    def __len__(self):
        return len(self.pre_data) if not self.only_2 else len(self.pre_data_only_2)

    def __getitem__(self, idx):
        """read a line of preprocessed open-retrieval quac file into a quac example"""
        entry = self.pre_data[idx] if not self.only_2 else self.pre_data_only_2[idx]

        return_feature_dict = {
            "id":       entry['id'],
            "question": entry['question'],
            "rewrite":  entry['rewrite'],
            "relevant_docs_pids": entry['relevant_docs_pids'],
            "irrelevant_docs_pids": entry['irrelevant_docs_pids'],
            "context":  entry['context'],
            "context_ids": entry['context_ids'],
            "context_topics": entry['context_topics'],
            "context_session": entry["context_session"],
        }

        return return_feature_dict
    
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on the model output using the attention mask.
    """
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_collate_fn(vs_context, bertopics, tokenizer, model_embedding, use_embed= True, only_question= False):
  device = model_embedding.device
  index_to_docstore_id = vs_context.index_to_docstore_id
  docstore_id_to_index = {y: x for x, y in index_to_docstore_id.items()}

  def collate_fn(batch):
      question_id = [i['id'] for i in batch]
      question_text = [i['question'] for i in batch]
      query_token_ids = tokenizer(question_text, padding=True, truncation=True, return_tensors='pt').to(device)

      relevant_docs_pids = [i['relevant_docs_pids'] for i in batch]
      irrelevant_docs_pids = [i['irrelevant_docs_pids'] for i in batch]

      if use_embed:
        if only_question:
          with torch.no_grad():
            embed_query = model_embedding(**query_token_ids)
          mean_query = mean_pooling(embed_query, query_token_ids['attention_mask'])

          re = {"question_id": question_id, "question_text": question_text, "relevant_docs_pids":relevant_docs_pids,"embed_query": embed_query[0], "mean_query": mean_query}
        else:
          batch_relevant = [torch.stack([torch.tensor(vs_context.index.reconstruct_n(docstore_id_to_index[j], 1)[0]) for j in i], dim=0).to(device) if len(i) != 0 else None for i in relevant_docs_pids]
          
          batch_irrelevant = torch.stack([torch.stack([torch.tensor(vs_context.index.reconstruct_n(docstore_id_to_index[int(j)], 1)[0]) for j in i], dim=0) for i in irrelevant_docs_pids], dim=0).to(device)
          
          context_topics = [i["context_topics"] for i in batch]
          batch_ctx = torch.stack([torch.Tensor(bertopics.topic_embeddings_[i]) for i in context_topics], dim=0).to(device)

          session = [i["context_session"] for i in batch]
          session_token_ids = tokenizer(session, padding=True, truncation=True, return_tensors='pt').to(device)

          with torch.no_grad():
            embed_query = model_embedding(**query_token_ids)
            embed_sess = model_embedding(**session_token_ids)

          mean_query = mean_pooling(embed_query, query_token_ids['attention_mask'])
          # batch_ctx = torch.stack([mean_pooling(embed_ctx[i], ctx_token_ids[i]['attention_mask']) for i in range(len(embed_ctx))], dim=0)
          # batch_relevant = [mean_pooling(embed_relevant[i], relevant_token_ids[i]['attention_mask'])  if embed_relevant[i] is not None else None for i in range(len(embed_relevant))]
          # batch_irrelevant = torch.stack([mean_pooling(embed_irrelevant[i], irrelevant_token_ids[i]['attention_mask']) for i in range(len(embed_irrelevant))], dim=0)
          if batch[0]['rewrite'] is not None:
              rewrite = [i['rewrite'] for i in batch]
              rewrite_token_ids =  tokenizer(rewrite, padding=True, truncation=True, return_tensors='pt').to(device)
              with torch.no_grad():
                embed_rewrite = model_embedding(**rewrite_token_ids)
              mean_rewrite = mean_pooling(embed_rewrite, rewrite_token_ids['attention_mask'])
          else: mean_rewrite = None

          re = {
                "question_id": question_id, "question_text": question_text, "embed_query":  embed_query[0], "mean_query": mean_query, "context": torch.cat([batch_ctx, embed_sess[0]], dim=1), "mean_context":batch_ctx.mean(dim=1), "rewrite": mean_rewrite,
                "relevant_docs_pids":relevant_docs_pids, "batch_relevant": batch_relevant, "batch_irrelevant": batch_irrelevant}
      else: re = {"question_id": question_id, "question_text": question_text, "relevant_docs_pids":relevant_docs_pids}
      return re
  return collate_fn

if __name__ == '__main__':
    train_file = "data/raw/train/new_train_conversation.json"
    test_file = "data/raw/test/new_test_conversation.json"
    collection_path = "data/raw/passage_corpus.json"
    bertopic_context = "data/bertopic/BerTopic_corpus_all-MiniLM-L6-v2"

    os.makedirs("data/processed", exist_ok=True)

    train_data = RetrieverDataset(train_file, history_num=2,bertopic_context_path=bertopic_context,collection_path=collection_path)
    test_dataset = RetrieverDataset(test_file,history_num=2, training=False,bertopic_context_path=bertopic_context,collection_path=collection_path)

    with open('data/processed/train_data_bertopic_all-MiniLM-L6-v2_negBM25.pkl', 'wb') as f:
      pickle.dump(train_data.pre_data, f)
    with open('data/processed/test_dataset_bertopic_all-MiniLM-L6-v2_negBM25.pkl', 'wb') as f:
      pickle.dump(test_dataset.pre_data, f)

