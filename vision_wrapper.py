import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
import io

from transformers import AutoProcessor
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaPreTrainedModel,
)

from transformers.models.qwen2_vl import (
    Qwen2VLForConditionalGeneration, 
    Qwen2VLConfig,
)

class ColPali(PaliGemmaPreTrainedModel):
    """
    ColPali model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    """
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
        # Delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)
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
        self.bs_query = 64
        self.model_name = "checkpoint/colpali-v1.1"
        self.base_ckpt = "checkpoint/colpaligemma-3b-mix-448-base"
        # Load model on cuda:0 by default!
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
        # Load processor
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

    def embed_quotes(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        embeddings = []
        dataloader = DataLoader(images, batch_size=self.bs, shuffle=False,
                                collate_fn=lambda x: self.process_images(x))
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="[ColPaliRetriever] Embedding images"):
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
        batch_docs = self.processor(
            text=texts,
            images=pil_images,
            return_tensors="pt",
            padding="longest"
        )
        return batch_docs

    def score(self, query_embs, image_embs):
        """
        Computes (batch) similarity scores MaxSim style.
        Inputs:
           query_embs: [Nq, seq, dim]
           image_embs: [Ni, seq, dim]
        Returns:
           scores: [Nq, Ni] max similarity per query-image (like ColBERT)
        """
        qs = [torch.from_numpy(e) for e in query_embs]
        ds = [torch.from_numpy(e) for e in image_embs]
        # MaxSim/colbert scoring: max dot product over sequence dimension
        # shape: [Q, D]
        scores = np.zeros((len(qs), len(ds)), dtype=np.float32)
        for i, q in enumerate(qs):
            q = q.float()  # [Lq, d]
            for j, d in enumerate(ds):
                d = d.float() # [Ld, d]
                # score = max_{q_token, d_token} q_token @ d_token.T
                sim = torch.matmul(q, d.T)    # [Lq, Ld]
                maxsim = torch.max(sim, dim=1)[0].sum().item()  # colbert-style batch: sum-of-max over query tokens
                scores[i, j] = maxsim
        return scores



class ColQwen2(Qwen2VLForConditionalGeneration):
    """
    ColQwen2 model implementation.
    """
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

        self.model = ColQwen2.from_pretrained(
            self.base_ckpt,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        self.model.load_adapter(self.model_name)
        self.model.eval()

        # DataParallel:
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
        
        # Resize and convert
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

    def process_queries(self, queries, max_length=50, suffix=None):
        if suffix is None:
            suffix = "<pad>" * 10
        texts_query = []
        for q in queries:
            q_ = f"Query: {q}{suffix}"
            texts_query.append(q_)
        batch_query = self.processor(
            text=texts_query,
            return_tensors="pt",
            padding="longest",
        )
        return batch_query

    def embed_queries(self, queries, pad=False):
        if isinstance(queries, str):
            queries = [queries]
        embeddings = []
        dataloader = DataLoader(
            queries, batch_size=self.bs_query, shuffle=False,
            collate_fn=lambda x: self.process_queries(x)
        )
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

    def embed_quotes(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        embeddings = []
        dataloader = DataLoader(
            images, batch_size=self.bs, shuffle=False,
            collate_fn=lambda x: self.process_images(x)
        )
        with torch.no_grad():
            dev = self.model.device_ids[0] if self.is_parallel else self.model.device
            for batch in tqdm(dataloader, desc="[ColQwen2Retriever] Embedding images"):
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
    
if __name__ == "__main__":

    retriever = ColPaliRetriever(bs=4)

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