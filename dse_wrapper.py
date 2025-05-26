# import torch
# from torch import nn
# from transformers import AutoModelForCausalLM

# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import numpy as np
# from PIL import Image
# from transformers import AutoProcessor


# class DSE(nn.Module):
#     """
#     DSE model wrapper for representation extraction.
#     """
#     def __init__(self, model_name, pooling: str = 'cls', normalize: bool = False,
#                  lora_adapter: str = None, device: str = "cpu"):
#         super().__init__()
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.bfloat16,
#             trust_remote_code=True,
#             attn_implementation="flash_attention_2",
#             use_cache=False,
#         )
#         self.model.config.pad_token_id = self.model.config.eos_token_id
#         if lora_adapter:
#             from peft import PeftModel
#             self.model = PeftModel.from_pretrained(self.model, lora_adapter)
#         self.pooling = pooling
#         self.normalize = normalize
#         # Move model to desired device
#         self.model = self.model.to(device)

#     def _pool(self, last_hidden_state, attention_mask):
#         if self.pooling in ['cls', 'first']:
#             reps = last_hidden_state[:, 0]
#         elif self.pooling in ['mean', 'avg', 'average']:
#             masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
#             reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
#         elif self.pooling in ['last', 'eos']:
#             sequence_lengths = attention_mask.sum(dim=1) - 1
#             batch_size = last_hidden_state.shape[0]
#             reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
#         else:
#             raise ValueError(f'Unknown pooling method: {self.pooling}')
#         if self.normalize:
#             reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
#         return reps

#     def encode_query(self, batch):
#         outputs = self.model(**batch, return_dict=True, output_hidden_states=True)
#         hs = outputs.hidden_states[-1]
#         reps = self._pool(hs, batch['attention_mask'])
#         return reps

#     def encode_passage(self, batch):
#         outputs = self.model(**batch, return_dict=True, output_hidden_states=True)
#         hs = outputs.hidden_states[-1]
#         reps = self._pool(hs, batch['attention_mask'])
#         return reps


# class DSERetriever:
#     def __init__(self, bs=4, use_gpu=True):
#         self.bs = bs
#         self.bs_query = 64
#         self.model_name = "checkpoint/dse-phi3-docmatix-v2"
#         # Set device properly
#         self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
#         # Pass device info to DSE!
#         self.model = DSE(self.model_name, device=self.device)
#         self.model.eval()
#         self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
#         self.mock_image = Image.new("RGB", (16, 16), color="black")

#     def process_queries(self, queries, max_length=128):
#         if isinstance(queries, str):
#             queries = [queries]
#         prompts = [f"query: {q}</s>" for q in queries]
#         batch = self.processor(
#             prompts,
#             images=None,
#             return_tensors="pt",
#             padding="longest",
#             max_length=max_length,
#             truncation=True,
#         )
#         return batch

#     def process_images(self, images, max_length=128):
#         pil_images = []
#         for img in images:
#             if isinstance(img, Image.Image):
#                 pil_img = img
#             elif isinstance(img, (bytes, bytearray)):
#                 from io import BytesIO
#                 pil_img = Image.open(BytesIO(img))
#             else:
#                 raise ValueError("Each image must be a PIL.Image.Image or bytes.")
#             pil_images.append(pil_img.convert("RGB"))
#         prompts = [f"<|image_{i+1}|>\nWhat is shown in this image?</s>" for i in range(len(pil_images))]
#         batch = self.processor(
#             prompts,
#             images=pil_images,
#             return_tensors="pt",
#             padding="longest",
#             max_length=max_length,
#             truncation=True,
#         )
#         if batch['input_ids'].dim() == 3:
#             batch['input_ids'] = batch['input_ids'].squeeze(0)
#             batch['attention_mask'] = batch['attention_mask'].squeeze(0)
#             if 'image_sizes' in batch:
#                 batch['image_sizes'] = batch['image_sizes'].squeeze(0)
#         return batch

#     def embed_queries(self, queries):
#         if isinstance(queries, str):
#             queries = [queries]
#         embeddings = []
#         dataloader = DataLoader(queries, batch_size=self.bs_query, shuffle=False, 
#                                 collate_fn=lambda x: self.process_queries(x))
#         with torch.no_grad():
#             for batch in tqdm(dataloader, desc="[DSERetriever] Embedding queries"):
#                 batch = {k: v.to(self.device) for k, v in batch.items()}
#                 q_emb = self.model.encode_query(batch)
#                 embeddings.extend(q_emb.cpu().float().numpy())
#         return embeddings

#     def embed_quotes(self, images):
#         if isinstance(images, (Image.Image, bytes, bytearray)):
#             images = [images]
#         embeddings = []
#         dataloader = DataLoader(images, batch_size=self.bs, shuffle=False,
#                                 collate_fn=lambda x: self.process_images(x))
#         with torch.no_grad():
#             for batch in tqdm(dataloader, desc="[DSERetriever] Embedding images"):
#                 batch = {k: v.to(self.device) for k, v in batch.items()}
#                 p_emb = self.model.encode_passage(batch)
#                 embeddings.extend(p_emb.cpu().float().numpy())
#         return embeddings

#     def score(self, query_embs, image_embs):
#         qs = np.stack(query_embs)
#         ds = np.stack(image_embs)
#         scores = np.matmul(qs, ds.T)
#         return scores
    

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

class DSE(nn.Module):
    """
    Merged DSE retrieval/embedding class: wraps model, handles encoding, pooling, processing and embedding.
    """
    def __init__(
        self, 
        model_name="checkpoint/dse-phi3-docmatix-v2",
        pooling='cls',
        normalize=False,
        lora_adapter=None,
        bs=4,
        bs_query=64,
        use_gpu=True,
    ):
        super().__init__()
        # Setup device
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
        # Load transformer model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # attn_implementation="flash_attention_2",
            use_cache=False,
        ).to(self.device)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        if lora_adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_adapter)
        # Pooling setup
        self.pooling = pooling
        self.normalize = normalize

        # Processor and utility
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.mock_image = Image.new("RGB", (16, 16), color="black")

        # Batching
        self.bs = bs
        self.bs_query = bs_query

    # ========== Pooling ==========
    def _pool(self, last_hidden_state, attention_mask):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'Unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    # ========== Input Preprocessing ==========
    def process_queries(self, queries, max_length=128):
        if isinstance(queries, str):
            queries = [queries]
        prompts = [f"query: {q}</s>" for q in queries]
        batch = self.processor(
            prompts,
            images=None,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        return batch

    def process_images(self, images, max_length=128):
        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_img = img
            elif isinstance(img, (bytes, bytearray)):
                from io import BytesIO
                pil_img = Image.open(BytesIO(img))
            else:
                raise ValueError("Each image must be a PIL.Image.Image or bytes.")
            pil_images.append(pil_img.convert("RGB"))
        prompts = [
            f"<|image_{i+1}|>\nWhat is shown in this image?</s>"
            for i in range(len(pil_images))
        ]
        batch = self.processor(
            prompts,
            images=pil_images,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        # Squeeze batch dims if needed
        if batch['input_ids'].dim() == 3:
            batch['input_ids'] = batch['input_ids'].squeeze(0)
            batch['attention_mask'] = batch['attention_mask'].squeeze(0)
            if 'image_sizes' in batch:
                batch['image_sizes'] = batch['image_sizes'].squeeze(0)
        return batch

    # ========== Embedding/Encoding ==========
    def encode_query(self, batch):
        outputs = self.model(**{k: v.to(self.device) for k, v in batch.items()},
                             return_dict=True, output_hidden_states=True)
        hs = outputs.hidden_states[-1]
        reps = self._pool(hs, batch['attention_mask'].to(self.device))
        return reps

    def encode_passage(self, batch):
        outputs = self.model(**{k: v.to(self.device) for k, v in batch.items()},
                             return_dict=True, output_hidden_states=True)
        hs = outputs.hidden_states[-1]
        reps = self._pool(hs, batch['attention_mask'].to(self.device))
        return reps

    def embed_queries(self, queries):
        if isinstance(queries, str):
            queries = [queries]
        embeddings = []
        dataloader = DataLoader(
            queries, batch_size=self.bs_query, shuffle=False,
            collate_fn=lambda xs: self.process_queries(xs)
        )
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="[DSE] Embedding queries"):
                reps = self.encode_query(batch)
                embeddings.extend(reps.cpu().float().numpy())
        return embeddings

    def embed_images(self, images):
        if isinstance(images, (Image.Image, bytes, bytearray)):
            images = [images]
        embeddings = []
        dataloader = DataLoader(
            images, batch_size=self.bs, shuffle=False,
            collate_fn=lambda xs: self.process_images(xs)
        )
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="[DSE] Embedding images"):
                reps = self.encode_passage(batch)
                embeddings.extend(reps.cpu().float().numpy())
        return embeddings

    # ========== Scoring ==========
    def score(self, query_embs, image_embs):
        qs = np.stack(query_embs)
        ds = np.stack(image_embs)
        scores = np.matmul(qs, ds.T)
        return scores



if __name__ == "__main__":

    retriever = DSE(bs=4)

    queries = ['how much protein should a female eat', 'how much food should a male eat or drink or shit']

    prefix = "/export/home2/zli/kc/minerU/MP-DocVQA/mineru_auto/ffdw0217/images/"
    images = [
        "1c6cdebf5de674caf4e21c79a7425b2b5da19d0db17f6b6df0e5d0de5a759eca.jpg",
        "3be268e0f904af72855eda91f1e9231a03d41b864acb9a551bad6f81519419b9.jpg",
        "5c77b31e0a6328bb35ec43bedee297e91584be96b347521c4e48f7ba4e3a85b2.jpg",
        "9c8947d6d05678a1768c2329e0b23a31d2dbc517f908524203ac20acd3f79aee.jpg",
        "75f737182bd5249f1066cf81fa7dc2281698e6a0421c24cb704610a708bba0e0.jpg",
        "0229e3e5ebad89660adfd8d578525d38e195ab7147c211bf49af0975a39837d1.jpg"
    ]

    # image_path = [prefix+x for x in images]
    images = [Image.open(prefix+x) for x in images]
    
    q_embeds = retriever.embed_queries(queries)
    for x in q_embeds:
        print(x.shape)

    q_embeds = retriever.embed_queries(queries, pad=True)
    for x in q_embeds:
        print(x.shape)

    img_embeds = retriever.embed_quotes(images)

    scores = retriever.score(q_embeds, img_embeds)
    print(scores)
    print(scores.argmax(axis=1))