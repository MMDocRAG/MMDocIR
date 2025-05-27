import torch
import numpy as np
from tqdm import tqdm

def is_str_list(obj): # Checks if it's a list and all elements are strings
    return isinstance(obj, list) and all(isinstance(item, str) for item in obj)

def is_np_list(obj): # Checks if it's a list and all elements are np.ndarray
    return isinstance(obj, list) and all(isinstance(item, np.ndarray) for item in obj)

def is_np_array(obj): # Checks if it's a np.ndarray
    return isinstance(obj, np.ndarray)

class Sent_Retriever:
    def __init__(self, bs=256, use_gpu=True):
        self.bs = bs
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")

    def embed_passages(self, passages, prefix=""):
        if prefix != "":
            passages = [prefix + item for item in passages]
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(passages), self.bs)):
                batch_passage = passages[i:(i + self.bs)]
                emb = self.model.encode(batch_passage, normalize_embeddings=True)
                embeddings.extend(emb)
        return embeddings

    def score(self, queries, quotes):
        if is_str_list(queries):
            query_emb = np.asarray(self.embed_queries(queries))
        elif is_np_list(queries):
            query_emb = np.asarray(queries)
        elif is_np_array(queries):
            query_emb = queries
        
        if is_str_list(quotes):
            quote_emb = np.asarray(self.embed_quotes(quotes))
        elif is_np_list(quotes):
            quote_emb = np.asarray(quotes)
        elif is_np_array(quotes):
            quote_emb = quotes

        return (query_emb @ quote_emb.T).tolist()

    def get_tok_len(self, text_input):
        return self.model._first_module().tokenizer(
            text=[text_input],
            truncation=False, max_length=False, return_tensors="pt"
        )["input_ids"].size()[-1]


class BGE(Sent_Retriever):
    def __init__(self, bs=256, use_gpu=True, model_path="checkpoint/bge-large-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        super().__init__(bs=bs, use_gpu=use_gpu)
        self.model_path = model_path
        self.model = SentenceTransformer(self.model_path)
        print("[text_wrapper.py - init] Setting up BGE...")
        print("[text_wrapper.py - init] BGE is loaded from '{}'...".format( self.model_path ))
        self.model.eval()
        self.model = self.model.to(self.device)

    def embed_queries(self, queries):
        prefix = "Represent this sentence for searching relevant passages："
        if isinstance(queries, str): queries = [queries]
        return self.embed_passages(queries, prefix)

    def embed_quotes(self, quotes):
        if isinstance(quotes, str): quotes = [quotes]
        return self.embed_passages(quotes)


class E5(Sent_Retriever):
    def __init__(self, bs=256, use_gpu=True, model_path="checkpoint/e5-large-v2"):
        from sentence_transformers import SentenceTransformer
        super().__init__(bs=bs, use_gpu=use_gpu)
        self.model_path = model_path
        self.model = SentenceTransformer(self.model_path)
        print("[text_wrapper.py - init] Setting up E5...")
        print("[text_wrapper.py - init] E5 is loaded from '{}'...".format( self.model_path ))
        self.model.eval()
        self.model = self.model.to(self.device)

    def embed_queries(self, queries):
        prefix = "query："
        if isinstance(queries, str): queries = [queries]
        return self.embed_passages(queries, prefix)

    def embed_quotes(self, quotes):
        prefix = "passage: "
        if isinstance(quotes, str): quotes = [quotes]
        return self.embed_passages(quotes, prefix)


class GTE(Sent_Retriever):
    def __init__(self, bs=256, use_gpu=True, model_path="checkpoint/gte-large"):
        from sentence_transformers import SentenceTransformer
        super().__init__(bs=bs, use_gpu=use_gpu)
        self.model_path = model_path
        self.model = SentenceTransformer(self.model_path)
        print("[text_wrapper.py - init] Setting up GTE...")
        print("[text_wrapper.py - init] GTE is loaded from '{}'...".format( self.model_path ))
        self.model.eval()
        self.model = self.model.to(self.device)

    def embed_queries(self, queries):
        if isinstance(queries, str): queries = [queries]
        return self.embed_passages(queries)

    def embed_quotes(self, quotes):
        if isinstance(quotes, str): quotes = [quotes]
        return self.embed_passages(quotes)


class Contriever():
    def __init__(self, bs = 256, use_gpu= True, model_path='checkpoint/contriever-msmarco'):
        from transformers import AutoTokenizer, AutoModel
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)
        self.bs = bs
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
        print("[text_wrapper.py - init] Setting up Contriever...")
        print("[text_wrapper.py - init] Contriever is loaded from '{}'...".format( self.model_path ))
        self.model.eval()
        self.model = self.model.to(self.device)

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def embed_queries(self, query):
        return self.embed_passages(query)

    def embed_quotes(self, quotes):
        return self.embed_passages(quotes)

    def embed_passages(self, quotes):
        if isinstance(quotes, str): quotes = [quotes]
        quote_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(quotes), self.bs)):
                batch_quotes = quotes[i:(i + self.bs)]
                encoded_quotes = self.tokenizer.batch_encode_plus(
                    batch_quotes, return_tensors = "pt",
                    max_length = 512, padding = True, truncation = True)
                encoded_data = {k: v.to(self.device) for k, v in encoded_quotes.items()}
                batched_outputs = self.model(**encoded_data)
                batched_quote_embs = self.mean_pooling(batched_outputs[0], encoded_data['attention_mask'])
                quote_embeddings.extend([q.cpu().detach().numpy() for q in batched_quote_embs])
        return quote_embeddings

    def score(self, queries, quotes):
        if is_str_list(queries):
            query_emb = np.asarray(self.embed_queries(queries))
        elif is_np_list(queries):
            query_emb = np.asarray(queries)
        elif is_np_array(queries):
            query_emb = queries
        
        if is_str_list(quotes):
            quote_emb = np.asarray(self.embed_quotes(quotes))
        elif is_np_list(quotes):
            quote_emb = np.asarray(quotes)
        elif is_np_array(quotes):
            quote_emb = quotes

        return (query_emb @ quote_emb.T).tolist()


class DPR():
    def __init__(self, bs = 256, use_gpu=True, model_path="checkpoint"):
        from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
        self.model_path = model_path
        self.query_tok = DPRQuestionEncoderTokenizer.from_pretrained(self.model_path +"/dpr-question_encoder-multiset-base")
        self.query_enc = DPRQuestionEncoder.from_pretrained(self.model_path +"/dpr-question_encoder-multiset-base")
        self.ctx_tok = DPRContextEncoderTokenizer.from_pretrained(self.model_path +"/dpr-ctx_encoder-multiset-base")
        self.ctx_enc = DPRContextEncoder.from_pretrained(self.model_path +"/dpr-ctx_encoder-multiset-base")
        self.bs = bs
        print("[text_wrapper.py - init] Setting up DPR...")
        print("[text_wrapper.py - init] DPR is loaded from '{}'...".format( self.model_path ))
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
        self.query_enc.eval()
        self.query_enc = self.query_enc.to(self.device)
        self.ctx_enc.eval()
        self.ctx_enc = self.ctx_enc.to(self.device)

    def embed_queries(self, queries):
        if isinstance(queries, str): queries = [queries]
        query_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(queries), self.bs)):
                batch_queries = queries[i:(i + self.bs)]
                encoded_query = self.query_tok.batch_encode_plus(
                    batch_queries, truncation=True, padding=True,
                    return_tensors='pt', max_length=512)
                encoded_data = {k : v.cuda() for k, v in encoded_query.items()}
                query_emb = self.query_enc(**encoded_data).pooler_output
                query_emb = [q.cpu().detach().numpy() for q in query_emb]
                query_embeddings.extend(query_emb)
        return query_embeddings

    def embed_quotes(self, quotes):
        if isinstance(quotes, str): quotes = [quotes]
        quote_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(quotes), self.bs)):
                batch_quotes = quotes[i:(i + self.bs)]
                encoded_ctx = self.ctx_tok.batch_encode_plus(
                    batch_quotes, truncation=True, padding=True,
                    return_tensors='pt', max_length=512)
                encoded_data = {k: v.cuda() for k, v in encoded_ctx.items()}
                quote_emb = self.ctx_enc(**encoded_data).pooler_output
                quote_emb = [q.cpu().detach().numpy() for q in quote_emb]
                quote_embeddings.extend(quote_emb)
        return quote_embeddings

    def score(self, queries, quotes):
        if is_str_list(queries):
            query_emb = np.asarray(self.embed_queries(queries))
        elif is_np_list(queries):
            query_emb = np.asarray(queries)
        elif is_np_array(queries):
            query_emb = queries
        
        if is_str_list(quotes):
            quote_emb = np.asarray(self.embed_quotes(quotes))
        elif is_np_list(quotes):
            quote_emb = np.asarray(quotes)
        elif is_np_array(quotes):
            quote_emb = quotes

        return (query_emb @ quote_emb.T).tolist()


class ColBERTReranker:
    def __init__(self, bs = 256, use_gpu= True, model_path="checkpoint/colbertv2.0"):
        from colbert.modeling.colbert import ColBERT
        from colbert.infra import ColBERTConfig
        from transformers import AutoTokenizer
        self.model_path = model_path
        self.bs = bs
        config = ColBERTConfig(bsize=bs, root='./', query_token_id='[Q]', doc_token_id='[D]')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = ColBERT(name=self.model_path, colbert_config=config)
        self.doc_token_id = self.tokenizer.convert_tokens_to_ids(config.doc_token_id)
        self.query_token_id = self.tokenizer.convert_tokens_to_ids(config.query_token_id)
        self.add_special_tokens = True
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
        print("[text_wrapper.py - init] Setting up ColBERT Reranker...")
        print("[text_wrapper.py - init] ColBERT is loaded from '{}'...".format( self.model_path ))
        self.model.eval()
        self.model = self.model.to(self.device)

    def embed_queries(self, queries):
        if isinstance(queries, str): queries = [queries]
        query_embeddings = []
        query = ['. ' + item for item in queries] # placeholder for query emb
        with torch.no_grad():
            for i in tqdm(range(0, len(queries), self.bs)):
                batch_queries = queries[i:(i + self.bs)]
                encoded_query = self.tokenizer.batch_encode_plus(
                    batch_queries, max_length = 512, padding=True, truncation=True,
                    add_special_tokens=self.add_special_tokens, return_tensors='pt')
                encoded_data = {k: v.to(self.device) for k, v in encoded_query.items()}
                encoded_data['input_ids'][:, 1] = self.query_token_id
                batch_query_emb = self.model.query(encoded_data['input_ids'], encoded_data['attention_mask'])

                for emb, mask in zip(batch_query_emb, encoded_data['attention_mask']):
                    length = mask.sum().item()  # Number of true tokens in this sequence
                    np_emb = emb[:length].cpu().numpy()  # Shape: [L, H]
                    query_embeddings.append(np_emb)      # `L` varies per example
        return query_embeddings

    @staticmethod
    def pad_tok_len(quote_embeddings, pad_value=0):
        lengths = [e.shape[0] for e in quote_embeddings]
        max_len = max(lengths)
        N, H = len(quote_embeddings), quote_embeddings[0].shape[1]
        padded_embeddings = np.full((N, max_len, H), pad_value, dtype=quote_embeddings[0].dtype)
        padded_masks = np.zeros((N, max_len), dtype=np.int64)
        for i, (emb, length) in enumerate(zip(quote_embeddings, lengths)):
            padded_embeddings[i, :length, :] = emb
            padded_masks[i, :length] = 1
        return padded_embeddings, padded_masks

    def embed_quotes(self, quotes, pad_token_len = False):
        quote_embeddings = []
        quote_masks = []
        quotes = ['. ' + quote for quote in quotes]
        with torch.no_grad():
            # placeholder for query emb
            for i in tqdm(range(0, len(quotes), self.bs)):
                batch_quotes = quotes[i:(i + self.bs)]
                encoded_quotes = self.tokenizer.batch_encode_plus(
                    batch_quotes, return_tensors = "pt",
                    max_length = 512, padding = True, truncation = True)
                encoded_data = {k: v.to(self.device) for k, v in encoded_quotes.items()}
                encoded_data['input_ids'][:, 1] = self.doc_token_id
                # bz x # max num_token in batch x 128
                batched_quote_embs = self.model.doc(encoded_data['input_ids'], encoded_data['attention_mask'])

                for emb, mask in zip(batched_quote_embs, encoded_data['attention_mask']):
                    length = mask.sum().item()  # Number of true tokens in this sequence
                    np_emb = emb[:length].cpu().numpy()  # Shape: [L, H]
                    quote_embeddings.append(np_emb)      # `L` varies per example

        # max length of quotes could differ between different batches
        if pad_token_len:
            quote_embeddings, quote_masks = self.pad_tok_len(quote_embeddings)
            return quote_embeddings, quote_masks
        return quote_embeddings

    @staticmethod
    def colbert_score(query_embed, quote_embeddings, quote_masks):
        Q, H = query_embed.shape # [Q, H]
        N, L, _ = quote_embeddings.shape # [N, L, H]
        query_expanded = query_embed[:, np.newaxis, np.newaxis, :]         # [Q, 1, 1, H]
        quote_expanded = quote_embeddings[np.newaxis, :, :, :]             # [1, N, L, H]
        sim = np.matmul(query_expanded, np.transpose(quote_expanded, (0 ,1 ,3 ,2)))  # (Q, N, 1, L)
        sim = np.einsum('qh,nlh->qnl', query_embed, quote_embeddings)      # [Q, N, L]
        sim = np.where(quote_masks[np.newaxis, :, : ]==1, sim, -1e9)   # Mask invalid tokens [Q, N, L]
        maxsim = sim.max(-1)   # MaxSim: For each query token, take max over quote tokens [Q, N]
        scores = maxsim.sum(axis=0)  # Aggregate (sum over query tokens) [N]
        return scores

    def score(self, queries, quotes):
        if is_str_list(queries):
            query_embed = self.embed_queries(queries)
        elif is_np_list(queries):
            query_embed = queries

        if is_str_list(quotes):
            quote_embed, quote_masks = self.embed_quotes(quotes, pad_token_len=True)
        elif is_np_list(quotes):
            quote_embed, quote_masks = self.pad_tok_len(quotes)

        scores_list = []
        for q_embed in query_embed:
            scores = self.colbert_score(q_embed, quote_embed, quote_masks)
            scores_list.append(scores.tolist())
        return scores_list
