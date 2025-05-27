import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
import io
import json

from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaPreTrainedModel,
)
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLConfig


class DSE(nn.Module):
    def __init__(self, model_name="checkpoint/dse-phi3-v1", lora_adapter=None, bs=4, flash_attn=True):
        super().__init__() # "checkpoint/dse-phi3-docmatix-v2" "checkpoint/dse-phi3-v1.0"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def set_attention_implementation(model_name, flash_attn):
            config_path = f"{model_name}/config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            config["_attn_implementation"] = "flash_attention_2" if flash_attn else None
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        set_attention_implementation(model_name, flash_attn)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, use_cache=False)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        if lora_adapter:
            self.model = self.model.load_adapter(lora_adapter)
        self.model = self.model.to(self.device)  # First move to primary GPU
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.bs = bs
        self.bs_query = 64

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
                reps = self.encode(batch)
                embeddings.extend(reps.cpu().float().numpy())
        return embeddings

    def embed_quotes(self, images, hybrid=False):
        if not hybrid:
            if isinstance(images, (Image.Image, bytes, bytearray)):
                images = [images]
            embeddings = []
            dataloader = DataLoader(
                images, batch_size=self.bs, shuffle=False,
                collate_fn=lambda xs: self.process_images(xs)
            )
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="[DSE] Embedding quotes in images"):
                    reps = self.encode(batch)
                    embeddings.extend(reps.cpu().float().numpy())
            return embeddings
        else: # input quotes in text format
            if isinstance(images, str):
                images = [images]
            embeddings = []
            dataloader = DataLoader(
                images, batch_size=self.bs_query, shuffle=False,
                collate_fn=lambda xs: self.process_image_texts(xs)
            )
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="[DSE] Embedding quotes in texts"):
                    reps = self.encode(batch)
                    embeddings.extend(reps.cpu().float().numpy())
            return embeddings
        
    def encode(self, batch):
        outputs = self.model(**{k: v.to(self.device) for k, v in batch.items()}, return_dict=True, output_hidden_states=True)
        hs = outputs.hidden_states[-1]
        reps = self._pool(hs, batch['attention_mask'].to(self.device))
        return reps

    def _pool(self, last_hidden_state, attention_mask):
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        return reps

    def process_queries(self, queries):
        if isinstance(queries, str):
            queries = [queries]
        prompts = [f"query: {q}</s>" for q in queries]
        batch = self.processor(
            prompts,
            images=None,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        return batch

    def process_image_texts(self, passages):
        if isinstance(passages, str):
            passages = [passages]
        prompts = [f"passage: {p}</s>" for p in passages]
        batch = self.processor(
            prompts,
            images=None,
            return_tensors="pt",
            padding="longest",
            max_length=600,
            truncation=True,
        )
        return batch
    
    def process_images(self, images):
        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_img = img.resize((1344, 1344))
            elif isinstance(img, (bytes, bytearray)):
                pil_img = Image.open(io.BytesIO(img))
                pil_img = pil_img.resize((1344, 1344))
            else:
                raise ValueError("Each image must be a PIL.Image.Image or bytes.")
            pil_images.append(pil_img.convert("RGB"))
        prompts = [f"<|image_{i+1}|>\nWhat is shown in this image?</s>" for i in range(len(pil_images))]
        batch = self.processor(
            prompts,
            images=pil_images,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        if batch['input_ids'].dim() == 3: # Squeeze batch dims if needed
            batch['input_ids'] = batch['input_ids'].squeeze(0)
            batch['attention_mask'] = batch['attention_mask'].squeeze(0)
            if 'image_sizes' in batch:
                batch['image_sizes'] = batch['image_sizes'].squeeze(0)
        return batch

    def score(self, query_embs, image_embs):
        query_emb = np.asarray(query_embs)
        quote_emb = np.asarray(image_embs)
        scores = (query_emb @ quote_emb.T).tolist()
        return scores


class ColPali(PaliGemmaPreTrainedModel):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__(config=config)
        model = PaliGemmaForConditionalGeneration(config=config)
        if model.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"model.language_model.{k}" for k in model.language_model._tied_weights_keys]
        self.model = model
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.config.text_config.hidden_size, self.dim)
        self.post_init()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        kwargs.pop("output_hidden_states", None) # Delete output_hidden_states from kwargs
        outputs = self.model(*args, output_hidden_states=True, **kwargs)  # (batch_size, sequence_length, hidden_size)
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)
        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)
        return proj


class ColPali(PaliGemmaPreTrainedModel):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__(config=config)
        model = PaliGemmaForConditionalGeneration(config=config)
        if model.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"model.language_model.{k}" for k in model.language_model._tied_weights_keys]
        self.model = model
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.config.text_config.hidden_size, self.dim)
        self.post_init()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        kwargs.pop("output_hidden_states", None) # Delete output_hidden_states from kwargs
        outputs = self.model(*args, output_hidden_states=True, **kwargs)  # (batch_size, sequence_length, hidden_size)
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)
        # L2 normalization
        proj = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)
        return proj


class ColPaliRetriever():
    def __init__(self, bs=4, use_gpu=True):
        self.bs = bs
        self.bs_query = 32
        self.model_name = "checkpoint/colpali-v1.1"
        self.base_ckpt = "checkpoint/colpaligemma-3b-mix-448-base"
        device = "cuda:0" if (torch.cuda.is_available() and use_gpu) else "cpu"
        self.model = ColPali.from_pretrained(
            self.base_ckpt, torch_dtype=torch.bfloat16, device_map=None  # <-- NONE: Don't use device_map
        )
        self.model.load_adapter(self.model_name)
        self.model = self.model.to(device)
        self.model.eval()
        # Multi-GPU with DataParallel
        if torch.cuda.device_count() > 1 and use_gpu:
            print(f"[ColPaliRetriever] Using DataParallel on {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device(device)
        print(f"[ColPaliRetriever - init] ColPali loaded from '{self.base_ckpt}' (Adapter '{self.model_name}')...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.mock_image = Image.new("RGB", (16, 16), color="black")

    def embed_queries(self, queries, pad=False):
        if isinstance(queries, str):
            queries = [queries]
        embeddings = []
        dataloader = DataLoader(queries, batch_size=self.bs_query, shuffle=False, 
                                collate_fn=lambda x: self.process_queries(x))
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="[ColPaliRetriever] Embedding queries"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                attention_mask = batch["attention_mask"]
                if isinstance(outputs, (tuple, list)): outputs = outputs[0]
                for emb, mask in zip(outputs, attention_mask):
                    if pad:
                        embeddings.append(emb.cpu().float().numpy())
                    else:
                        emb_nonpad = emb[mask.bool()]
                        embeddings.append(emb_nonpad.cpu().float().numpy())
        return embeddings

    def embed_quotes(self, images, hybrid=False):
        if not hybrid:
            if isinstance(images, Image.Image):
                images = [images]
            embeddings = []
            dataloader = DataLoader(images, batch_size=self.bs, shuffle=False,
                                    collate_fn=lambda x: self.process_images(x))
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="[ColPaliRetriever] Embedding quotes in images"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    for emb in torch.unbind(outputs):
                        embeddings.append(emb.cpu().float().numpy())
            return embeddings
        else: # input quotes in text format
            if isinstance(images, str):
                images = [images]
            embeddings = []
            dataloader = DataLoader(images, batch_size=self.bs, shuffle=False,
                                    collate_fn=lambda x: self.process_image_texts(x))
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="[ColPaliRetriever] Embedding quotes in texts"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    for emb in torch.unbind(outputs):
                        embeddings.append(emb.cpu().float().numpy())
            return embeddings

    def process_queries(self, queries, max_length=512):
        texts_query = [f"Question: {q}" + "<pad>" * 10 for q in queries]
        sl = getattr(self.processor, "image_seq_length", 32) # 1024
        batch_query = self.processor(
            images=[self.mock_image] * len(texts_query),
            text=texts_query,
            return_tensors="pt",
            padding="longest",
            max_length=max_length + sl   # fallback seq len
        )
        if "pixel_values" in batch_query: del batch_query["pixel_values"]
        batch_query["input_ids"] = batch_query["input_ids"][..., sl :]
        batch_query["attention_mask"] = batch_query["attention_mask"][..., sl :]
        return batch_query

    def process_image_texts(self, passages, max_length=600):
        def truncate_passages(passages, max_words=400):
            truncated = []
            for passage in passages:
                words = passage.split()
                if len(words) > max_words:
                    truncated_passage = ' '.join(words[:max_words])
                else:
                    truncated_passage = passage
                truncated.append(truncated_passage)
            return truncated
        passages = truncate_passages(passages)
        texts_passage = [f"Passage: {p}" + "<pad>" * 10 for p in passages]
        sl = getattr(self.processor, "image_seq_length", 32) # 1024
        batch_passage = self.processor(
            images=[self.mock_image] * len(texts_passage),
            text=texts_passage,
            return_tensors="pt",
            padding="longest",
            max_length=max_length + sl   # fallback seq len
        )
        if "pixel_values" in batch_passage: del batch_passage["pixel_values"]
        batch_passage["input_ids"] = batch_passage["input_ids"][..., sl :]
        batch_passage["attention_mask"] = batch_passage["attention_mask"][..., sl :]
        return batch_passage
    
    def process_images(self, images): 
        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):  # Already a PIL Image
                pil_img = img
            elif isinstance(img, (bytes, bytearray)):  # Binary image (e.g., from buffered.getvalue())
                pil_img = Image.open(io.BytesIO(img))
            else:
                raise ValueError("Each image must be a PIL.Image.Image or bytes.")
            pil_images.append(pil_img.convert("RGB"))
        
        texts = ["Describe the image."] * len(pil_images)
        batch_docs = self.processor(text=texts, images=pil_images,  return_tensors="pt", padding="longest")
        return batch_docs

    def score(self, query_embs, image_embs):
        qs = [torch.from_numpy(e) for e in query_embs]
        ds = [torch.from_numpy(e) for e in image_embs]
        # MaxSim/colbert scoring: max dot product over sequence dimension
        scores = np.zeros((len(qs), len(ds)), dtype=np.float32)
        for i, q in enumerate(qs):
            q = q.float()  # [Lq, d]
            for j, d in enumerate(ds):
                d = d.float() # [Ld, d]
                sim = torch.matmul(q, d.T)    # [Lq, Ld]
                maxsim = torch.max(sim, dim=1)[0].sum().item()  # colbert-style batch: sum-of-max over query tokens
                scores[i, j] = maxsim
        return scores


class ColQwen2(Qwen2VLForConditionalGeneration):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)
        self.dim = 128
        self.custom_text_proj = torch.nn.Linear(self.model.config.hidden_size, self.dim)
        self.padding_side = "left"
        self.post_init()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        kwargs.pop("output_hidden_states", None)
        # scatter hack for DDP, see original code if needed
        if "pixel_values" in kwargs and "image_grid_thw" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
            kwargs["pixel_values"] = torch.cat([pv[:o] for pv, o in zip(kwargs["pixel_values"], offsets)], dim=0)

        position_ids, rope_deltas = self.get_rope_index(
            input_ids=kwargs["input_ids"],
            image_grid_thw=kwargs.get("image_grid_thw", None),
            video_grid_thw=None,
            attention_mask=kwargs.get("attention_mask", None),
        )
        outputs = super().forward(*args,
                                  **kwargs,
                                  position_ids=position_ids,
                                  rope_deltas=rope_deltas,
                                  use_cache=False,
                                  output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        proj = self.custom_text_proj(last_hidden_states)
        proj = proj / proj.norm(dim=-1, keepdim=True)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)
        return proj


class ColQwen2Retriever:
    def __init__(self, bs=4, use_gpu=True):
        self.bs = bs
        self.bs_query = 64
        self.model_name = "checkpoint/colqwen2-v1.0"
        self.base_ckpt = "checkpoint/colqwen2-base"
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.model = ColQwen2.from_pretrained(self.base_ckpt, torch_dtype=torch.bfloat16, device_map=self.device)
        self.model.load_adapter(self.model_name)
        self.model.eval()

        self.is_parallel = False
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = torch.nn.DataParallel(self.model)
            self.is_parallel = True

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.min_pixels = 4 * 28 * 28
        self.max_pixels = 768 * 28 * 28
        self.factor = 28
        self.max_ratio = 200

    # ---------- Image Processing Utilities ----------
    @staticmethod
    def round_by_factor(number, factor):
        return round(number / factor) * factor

    @staticmethod
    def ceil_by_factor(number, factor):
        return math.ceil(number / factor) * factor

    @staticmethod
    def floor_by_factor(number, factor):
        return math.floor(number / factor) * factor

    def smart_resize(self, height: int, width: int) -> tuple:
        if max(height, width) / min(height, width) > self.max_ratio:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {self.max_ratio}, "
                f"got {max(height, width) / min(height, width)}"
            )
        h_bar = max(self.factor, self.round_by_factor(height, self.factor))
        w_bar = max(self.factor, self.round_by_factor(width, self.factor))
        if h_bar * w_bar > self.max_pixels:
            beta = math.sqrt((height * width) / self.max_pixels)
            h_bar = self.floor_by_factor(height / beta, self.factor)
            w_bar = self.floor_by_factor(width / beta, self.factor)
        elif h_bar * w_bar < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, self.factor)
            w_bar = self.ceil_by_factor(width * beta, self.factor)
        return h_bar, w_bar

    def process_images(self, images):
        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_img = img
            elif isinstance(img, (bytes, bytearray)):
                pil_img = Image.open(io.BytesIO(img))
            else:
                raise ValueError("Each image must be a PIL.Image.Image or bytes.")
            pil_images.append(pil_img.convert("RGB"))
        
        resized_images = []
        for image in pil_images:
            orig_size = image.size
            resized_height, resized_width = self.smart_resize(orig_size[1], orig_size[0])
            out_img = image.resize((resized_width,resized_height)).convert('RGB')
            resized_images.append(out_img)

        texts_doc = [ 
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n"
        ] * len(resized_images)

        batch_doc = self.processor(
            text=texts_doc,
            images=resized_images,
            padding="longest",
            return_tensors="pt"
        )
        # The following hack can be skipped during inference unless you run into shape mismatch
        offsets = batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]
        pixel_values = torch.split(batch_doc["pixel_values"], offsets.tolist())
        max_length = max([len(pv) for pv in pixel_values])
        pixel_values = [torch.cat([pv,
                                   torch.zeros((max_length - len(pv), pv.shape[1]),
                                               dtype=pv.dtype, device=pv.device)]) for pv in pixel_values]
        batch_doc["pixel_values"] = torch.stack(pixel_values)
        return batch_doc

    def process_queries(self, queries, max_length=600, suffix=None):
        if suffix is None:
            suffix = "<pad>" * 10
        texts_query = []
        for q in queries:
            q_ = f"Query: {q}{suffix}"
            texts_query.append(q_)
        batch_query = self.processor(text=texts_query, return_tensors="pt", padding="longest", max_length=600)
        return batch_query


    def process_image_texts(self, passages, max_length=600, suffix=None):
        if suffix is None:
            suffix = "<pad>" * 10
        texts_passage = []
        for p in passages:
            p_ = f"Passage: {p}{suffix}"
            texts_passage.append(p_)
        batch_passage = self.processor(text=texts_passage, return_tensors="pt", padding="longest", max_length=600)
        return batch_passage
    

    def embed_queries(self, queries, pad=False):
        if isinstance(queries, str):
            queries = [queries]
        embeddings = []
        dataloader = DataLoader(
            queries, batch_size=self.bs_query, shuffle=False,
            collate_fn=lambda x: self.process_queries(x))
        with torch.no_grad():
            # Use main device for DataParallel
            dev = self.model.device_ids[0] if self.is_parallel else self.model.device
            for batch in tqdm(dataloader, desc="[ColQwen2Retriever] Embedding queries"):
                batch = {k: v.to(dev) for k, v in batch.items()}
                outputs = self.model(**batch)
                attention_mask = batch["attention_mask"]
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                for emb, mask in zip(outputs, attention_mask):
                    if pad:
                        embeddings.append(emb.cpu().float().numpy())
                    else:
                        emb_nonpad = emb[mask.bool()]
                        embeddings.append(emb_nonpad.cpu().float().numpy())
        return embeddings
    

    def embed_quotes(self, images, hybrid=False):
        if not hybrid:
            if isinstance(images, Image.Image):
                images = [images]
            embeddings = []
            dataloader = DataLoader(
                images, batch_size=self.bs, shuffle=False,
                collate_fn=lambda x: self.process_images(x))
            with torch.no_grad():
                dev = self.model.device_ids[0] if self.is_parallel else self.model.device
                for batch in tqdm(dataloader, desc="[ColQwen2Retriever] Embedding quotes in images"):
                    batch = {k: v.to(dev) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    for emb in torch.unbind(outputs):
                        embeddings.append(emb.cpu().float().numpy())
            return embeddings
        else: # input quotes in text format
            if isinstance(images, str):
                images = [images]
            embeddings = []
            dataloader = DataLoader(
                images, batch_size=self.bs, shuffle=False,
                collate_fn=lambda x: self.process_image_texts(x)
            )
            with torch.no_grad():
                dev = self.model.device_ids[0] if self.is_parallel else self.model.device
                for batch in tqdm(dataloader, desc="[ColQwen2Retriever] Embedding quotes in texts"):
                    batch = {k: v.to(dev) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    for emb in torch.unbind(outputs):
                        embeddings.append(emb.cpu().float().numpy())
            return embeddings
    
    def score(self, query_embs, image_embs):
        qs = [torch.from_numpy(e) for e in query_embs]
        ds = [torch.from_numpy(e) for e in image_embs]
        scores = np.zeros((len(qs), len(ds)), dtype=np.float32)
        for i, q in enumerate(qs):
            q = q.float()
            for j, d in enumerate(ds):
                d = d.float()
                sim = torch.matmul(q, d.T)
                maxsim = torch.max(sim, dim=1)[0].sum().item()
                scores[i, j] = maxsim
        return scores
