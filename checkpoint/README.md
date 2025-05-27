## Download Retriever Checkpoints 

Download relavent retrievers (either text or visual retrievers) from huggingface: [MMDocIR/MMDocIR_Retrievers](https://huggingface.co/MMDocIR/MMDocIR_Retrievers). The list of available retrievers are as follows:

- **BGE**: [bge-large-en-v1.5](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/bge-large-en-v1.5) which is cloned from [BAAI](https://huggingface.co/BAAI)/[bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5).
- **ColBERT**: [colbertv2.0](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/colbertv2.0) which is cloned from [colbert-ir](https://huggingface.co/colbert-ir)/[colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0).
- **E5**: [e5-large-v2](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/e5-large-v2) which is cloned from [ intfloat](https://huggingface.co/intfloat)/[e5-large-v2](https://huggingface.co/intfloat/e5-large-v2).
- **GTE**: [gte-large](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/gte-large) which is cloned from [thenlper](https://huggingface.co/thenlper)/[gte-large](https://huggingface.co/thenlper/gte-large).
- **Contriever**: [contriever-msmarco](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/contriever-msmarco) which is cloned from [facebook](https://huggingface.co/facebook)/[contriever-msmarco](https://huggingface.co/facebook/contriever-msmarco).
- **DPR**:
  - question encoder: [dpr-question_encoder-multiset-base](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/dpr-question_encoder-multiset-base) which is cloned from [facebook](https://huggingface.co/facebook)/[dpr-question_encoder-multiset-base](https://huggingface.co/facebook/dpr-question_encoder-multiset-base).
  - passage encoder: [dpr-ctx_encoder-multiset-base](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/dpr-ctx_encoder-multiset-base) which is cloned from [facebook](https://huggingface.co/facebook)/[dpr-ctx_encoder-multiset-base](https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base).
- **ColPali**:
  - retriever adapter: [colpali-v1.1](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/colpali-v1.1) which is cloned from [vidore](https://huggingface.co/vidore)/[colpali-v1.1](https://huggingface.co/vidore/colpali-v1.1).
  - retriever base VLM: [colpaligemma-3b-mix-448-base](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/colpaligemma-3b-mix-448-base) which is cloned from [vidore](https://huggingface.co/vidore)/[colpaligemma-3b-mix-448-base](https://huggingface.co/vidore/colpaligemma-3b-mix-448-base).
- **ColQwen**:
  - retriever adapter: [colqwen2-v1.0](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/colqwen2-v1.0) which is cloned from [vidore](https://huggingface.co/vidore)/[colqwen2-v1.0](https://huggingface.co/vidore/colqwen2-v1.0).
  - retriever base VLM: [colqwen2-base](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/colqwen2-base) which is cloned from [vidore](https://huggingface.co/vidore)/[colqwen2-base](https://huggingface.co/vidore/colqwen2-base).
- **DSE-wikiss**: [dse-phi3-v1](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/dse-phi3-v1) which is processed as follows:
  -  clone from [Tevatron](https://huggingface.co/Tevatron)/[dse-phi3-v1.0](https://huggingface.co/Tevatron/dse-phi3-v1.0).
  -  fix  batch processing issue based on: https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/discussions/32/files
  -  change `config.json` and `preprocessor_config.json` to point to .py files in checkpoint.
  
- **DSE-docmatix**: [dse-phi3-docmatix-v2](https://huggingface.co/MMDocIR/MMDocIR_Retrievers/tree/main/dse-phi3-docmatix-v2) which is cloned from [Tevatron](https://huggingface.co/Tevatron)/[dse-phi3-docmatix-v2](https://huggingface.co/Tevatron/dse-phi3-docmatix-v2).




## Environment

```bash
python 3.9
torch2.4.0+cu121
transformers==4.45.0
sentence-transformers==2.2.2  # for BGE, GTE, E5 retrievers
colbert-ai==0.2.21 # for colbert retriever
flash-attn==2.7.4.post1  # for DSE retrievers to run with flash attention
```



## How to use these checkpoints

We standardize codes for all retrievers in two python files

- **For text retrievers**: refer to [`text_wrapper.py`](https://github.com/MMDocRAG/MMDocIR/blob/main/text_wrapper.py)
- **For vision retrievers**: refer to [`vision_wrapper.py`](https://github.com/MMDocRAG/MMDocIR/blob/main/vision_wrapper.py)

If you want to encode [MMDocIR_Evaluation_Dataset](https://huggingface.co/datasets/MMDocIR/MMDocIR_Evaluation_Dataset) with these retrievers, you can refer to code [MMDocIR](https://github.com/MMDocRAG/MMDocIR/tree/main)/[encode.py](https://github.com/MMDocRAG/MMDocIR/blob/main/encode.py) and [inference command](https://github.com/MMDocRAG/MMDocIR?tab=readme-ov-file#3-inference-command).

If you want to encode your own queries/pages/layouts with these retrievers, some simple demo codes are:

- **For text retrievers**:

  ```python
  From text_wrapper import DPR, BGE, GTE, E5, ColBERTReranker, Contriever
  
  retriever = E5()
  query = ['how much protein should a child consume', 'What is the CDC requirements for women?']
  passage = [
      "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
      "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
      "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
  ]
  scores = retriever.score(query, passage)
  print(scores)
  ```

- **For image retrievers**:

  ```python
  From vision_wrapper import DSE, ColQwen2Retriever, ColPaliRetriever
  
  retriever = DSE(model_name="checkpoint/dse-phi3-v1", bs=2)
  
  query = ['how much protein should a child consume', 'What is the CDC requirements for women?']
  prefix = "/home/user/xxx"
  images = [
      "0704.0418_1.jpg",
      "0704.0418_2.jpg",
      "0704.0418_3.jpg",
      "0705.1104_0.jpg",
      "0705.1104_1.jpg",
      "0705.1104_2.jpg",
      "0704.0418_1.jpg",
      "0704.0418_2.jpg",
      "0704.0418_3.jpg",
      "0705.1104_0.jpg",
      "0705.1104_1.jpg",
      "0705.1104_2.jpg",
  ]
  images = [Image.open(prefix+x) for x in images]
  q_embeds = retriever.embed_queries(queries)
  img_embeds = retriever.embed_quotes(images)
  
  scores = retriever.score(q_embeds, img_embeds)
  print(scores)
  ```




## Environment

```bash
python 3.9
torch2.4.0+cu121
transformers==4.45.0
sentence-transformers==2.2.2
colbert-ai==0.2.21
```
