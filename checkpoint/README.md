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
- **DSE-wikiss** which is cloned from [Tevatron](https://huggingface.co/Tevatron)/[dse-phi3-v1.0](https://huggingface.co/Tevatron/dse-phi3-v1.0).
- **DSE-docmatix** which is cloned from [Tevatron](https://huggingface.co/Tevatron)/[dse-phi3-docmatix-v2](https://huggingface.co/Tevatron/dse-phi3-docmatix-v2).




## Environment

```bash
python 3.9
torch2.4.0+cu121
transformers==4.45.0
sentence-transformers==2.2.2
colbert-ai==0.2.21
```
