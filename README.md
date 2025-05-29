<p align="center">
  <h1 align="center">MMDocIR: Benchmarking Multi-Modal Retrieval for Long Documents</h1>
    <p align="center">
    <strong>Kuicai Dong*</strong></a>
    ·
    <strong>Yujing Chang*</strong>
    ·
    <strong>Derrick Xin Deik Goh*</strong>
    ·
    <strong>Dexun Li</strong>
    ·
    <a href="https://scholar.google.com/citations?user=fUtHww0AAAAJ&hl=en"><strong>Ruiming Tang</strong></a>
    ·
    <a href="https://stephenliu0423.github.io/"><strong>Yong Liu</strong></a>
  </p>
<p align="center">
    📖<a href="https://arxiv.org/abs/2501.08828">Paper</a> |
    🏠<a href="https://mmdocrag.github.io/MMDocIR/">Homepage</a> |
    🤗<a href="https://huggingface.co/MMDocIR">Huggingface</a> |
	👉<a href="https://github.com/mmdocrag/MMDocIR">Github</a>
</p>
<p align="left">
  <p>
Multimodal document retrieval aims to identify and retrieve various forms of multimodal content, such as figures, tables, charts, and layout information from extensive documents. Despite its increasing popularity, there is a notable lack of a comprehensive and robust benchmark to effectively evaluate the performance of systems in such tasks. To address this gap, this work introduces a new benchmark, named MMDocIR, that encompasses two distinct tasks: page-level and layout-level retrieval. The former evaluates the performance of identifying the most relevant pages within a long document, while the later assesses the ability of detecting specific layouts, providing a more fine-grained measure than whole-page analysis. A layout
refers to a variety of elements, including textual paragraphs, equations, figures, tables, or charts. The MMDocIR benchmark comprises a rich dataset featuring 1,685 questions annotated by experts and 173,843 questions with bootstrapped labels, making it a valuable resource in multimodal document retrieval for
both training and evaluation. Through rigorous experiments, we demonstrate that (i) visual retrievers significantly outperform their text counterparts, (ii) MMDocIR training set effectively enhances the performance of multimodal document retrieval and (iii) text retrievers leveraging VLM-text significantly outperforms retrievers relying on OCR-text.
  </p>
  <a href="">
    <img src="static/images/top_figure1.png" alt="Logo" width="100%">
  </a>
<br>






## 🔮Evalulation Dataset

### 1. Download Datasets 

Download [`MMDocIR_pages.parquet`](https://huggingface.co/datasets/MMDocIR/MMDocIR_Evaluation_Dataset/blob/main/MMDocIR_pages.parquet) and [`MMDocIR_layouts.parquet`](https://huggingface.co/datasets/MMDocIR/MMDocIR_Evaluation_Dataset/blob/main/MMDocIR_layouts.parquet) from huggingface: [MMDocIR/MMDocIR_Evaluation_Dataset](https://huggingface.co/datasets/MMDocIR/MMDocIR_Evaluation_Dataset)

Place two parquet files under [`./dataset/`](https://github.com/MMDocRAG/MMDocIR/tree/main/dataset)



### 2. Download Retriever Checkpoints 

Download relavent retrievers (either text or visual retrievers) from huggingface: [MMDocIR/MMDocIR_Retrievers](https://huggingface.co/MMDocIR/MMDocIR_Retrievers).

**For text retrievers**:
- **BGE**: [bge-large-en-v1.5](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/bge-large-en-v1.5)
- **ColBERT**: [colbertv2.0](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/colbertv2.0)
- **E5**: [e5-large-v2](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/e5-large-v2)
- **GTE**: [gte-large](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/gte-large)
- **Contriever**: [contriever-msmarco](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/contriever-msmarco)
- **DPR**
  - question encoder: [dpr-question_encoder-multiset-base](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/dpr-question_encoder-multiset-base)
  - passage encoder: [dpr-ctx_encoder-multiset-base](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/dpr-ctx_encoder-multiset-base)

**For visual retrievers**:
- **ColPali**
  - retriever adapter: [colpali-v1.1](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/colpali-v1.1)
  - retriever base VLM: [colpaligemma-3b-mix-448-base](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/colpaligemma-3b-mix-448-base)
- **ColQwen**
  - retriever adapter: [colqwen2-v1.0](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/colqwen2-v1.0)
  - retriever base VLM: [colqwen2-base](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/colqwen2-base)
- **DSE-wikiss**: [dse-phi3-v1](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/dse-phi3-v1)
- **DSE-docmatix**: [dse-phi3-docmatix-v2](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/dse-phi3-docmatix-v2)

Place these checkpoints under [`./checkpoint/`](https://github.com/MMDocRAG/MMDocIR/tree/main/checkpoint)

#### Environment

```bash
python 3.9
torch2.4.0+cu121
transformers==4.45.0
sentence-transformers==2.2.2   # for BGE, GTE, E5 retrievers
colbert-ai==0.2.21             # for colbert retriever
flash-attn==2.7.4.post1        # for DSE retrievers with flash attention
```



### 3. Inference Command

You can infer using the command:

```bash
python encode.py BGE --bs 256 --mode vlm_text --encode query,page,layout
```

>`model` : the model name for example "BGE", is compulsory. All available models are `["BGE", "E5", "GTE", "Contriever", "DPR", "ColBERT", "ColPali", "ColQwen", "DSE-docmatix", "DSE-wikiss"]`
>
>`--mode` parameter (`choices=['vlm_text', 'oct_text', 'image_binary', 'image_hybrid'], default='vlm_text'`) is to control passing pages or layouts as either vlm_text, ocr_text, image_binary, image_hybrid. 
>
>`--encode` parameter (`default="query,page,layout"`) is by default encode for all queries, pages, and layouts. 
>
>- You can encode select any of [query, page, layout] and use `,` to seperate. 
>- For example:  encoding query and page is via `--encode query,page` , encoding page only is via `--encode page`.



### 4. For evaluation

You can infer using the command:

```bash
python search.py BGE --encode page,layout --encode_path encode
```

>`model` : the model name for example "BGE", is compulsory. All available models are `["BGE", "E5", "GTE", "Contriever", "DPR", "ColBERT", "ColPali", "ColQwen", "DSE-docmatix", "DSE-wikiss"]`
>
>`--encode` parameter (`default="page,layout"`) is by default score topk recall for page-level and layout-level. 
>
>- You can obtain only page-level scores via `--encode page` ,  or layout-level scores via `--encode layout`.
>
>`--encode_path` parameter (`default="encode"`)  to indicate the stored embedding of query, page, and layout.
>
>- For example, to score BGE results, by default we look for 3 pikle files named:
>  -  `./encode/encoded_query_BGE.pkl`
>  -  `./encode/encoded_page_BGE.pkl`
>  -  `./encode/encoded_layout_BGE.pkl`





## 🛠️Training Dataset

### 1. Download Datasets 

Download  all [parquet](https://huggingface.co/datasets/MMDocIR/MMDocIR_Train_Dataset/tree/main/parquet) and [json line](https://huggingface.co/datasets/MMDocIR/MMDocIR_Train_Dataset/tree/main/annotations_top1_negative) files from huggingface: [MMDocIR/MMDocIR_Train_Dataset](https://huggingface.co/datasets/MMDocIR/MMDocIR_Train_Dataset)



## 2. Dataset Class

Refer to [train_dataset.py](https://huggingface.co/datasets/MMDocIR/MMDocIR_Train_Dataset/blob/main/train_dataset.py)



### 3. Training Code

Coming soon






## 💾Citation
```
@misc{dong2025mmdocirbenchmarkingmultimodalretrieval,
      title={MMDocIR: Benchmarking Multi-Modal Retrieval for Long Documents}, 
      author={Kuicai Dong and Yujing Chang and Xin Deik Goh and Dexun Li and Ruiming Tang and Yong Liu},
      year={2025},
      eprint={2501.08828},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2501.08828}, 
}
```



## 📄 License

![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and code are intended and licensed for research use only.
License: Attribution-NonCommercial 4.0 International It should abide by the policy of OpenAI: https://openai.com/policies/terms-of-use
