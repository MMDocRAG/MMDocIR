import pandas as pd
import json
import pickle
import argparse


def get_queries(file_in):
    query_list, query_indices = [], []
    q_count = 0
    for line in open(file_in, 'r', encoding="utf-8"):
        item = json.loads(line.strip())
        doc_page = item["page_indices"]
        doc_layout = item["layout_indices"]
        for qa in item["questions"]:
            query_list.append(qa["Q"])
            # tuple of question index, start/end indices of doc
            query_indices.append((q_count, *doc_page, *doc_layout))
            q_count += 1
    return query_list, query_indices


def get_pages(file_in, mode="vlm_text"):
    q_list, q_indices = [], []
    dataset_df = pd.read_parquet(file_in)
    for row_index, row in dataset_df.iterrows():
        q_list.append(row[mode])
        q_indices.append(row_index)
    return q_list, q_indices


def get_layouts(file_in, mode="vlm_text"):
    q_list, q_indices = [], []
    dataset_df = pd.read_parquet(file_in)
    for row_index, row in dataset_df.iterrows():
        layout_type = row["type"]
        bbox = row["bbox"]
        page_id = row["page_id"]
        # page_size = row["page_size"]
        if mode == "image_binary":
            q_list.append(row["image_binary"])
        else:
            if layout_type in ["table", "image"]: q_list.append(row[mode])
            else: q_list.append(row["text"])
        q_indices.append((row_index, page_id, *bbox))
    return q_list, q_indices


def get_layouts_hybrid(file_in):
    q_img_list, q_img_indices = [], []
    q_txt_list, q_txt_indices = [], []
    dataset_df = pd.read_parquet(file_in)
    for row_index, row in dataset_df.iterrows():
        layout_type = row["type"]
        bbox = row["bbox"]
        page_id = row["page_id"]
        if layout_type in ["table", "image"]: 
            q_img_list.append(row["image_binary"])
            q_img_indices.append((row_index, page_id, *bbox))
        else:
            q_txt_list.append(row["text"])
            q_txt_indices.append((row_index, page_id, *bbox))
    return q_img_list, q_img_indices, q_txt_list, q_txt_indices


def get_retriever(model, bs):
    if model == "BGE":
        from text_wrapper import BGE
        bs = bs if bs != -1 else 256
        return BGE(bs=bs)
    elif model == "E5":
        from text_wrapper import E5
        bs = bs if bs != -1 else 256
        return E5(bs=bs)
    elif model == "GTE":
        from text_wrapper import GTE
        bs = bs if bs != -1 else 256
        return GTE(bs=bs)
    elif model == "Contriever":
        from text_wrapper import Contriever
        bs = bs if bs != -1 else 256
        return Contriever(bs=bs)
    elif model == "DPR":
        from text_wrapper import DPR
        bs = bs if bs != -1 else 256
        return DPR(bs=bs)
    elif model == "ColBERT":
        from text_wrapper import ColBERTReranker
        bs = bs if bs != -1 else 256
        return ColBERTReranker(bs=bs)

    elif model == "ColPali":
        from vision_wrapper import ColPaliRetriever
        bs = bs if bs != -1 else 10
        return ColPaliRetriever(bs=bs)

    elif model == "ColQwen":
        from vision_wrapper import ColQwen2Retriever
        bs = bs if bs != -1 else 8
        return ColQwen2Retriever(bs=bs)

    elif model == "DSE-docmatix":
        from vision_wrapper import DSE
        bs = bs if bs != -1 else 2
        return DSE(model_name="checkpoint/dse-phi3-docmatix-v2", bs=bs)

    elif model == "DSE-wikiss":
        from vision_wrapper import DSE
        bs = bs if bs != -1 else 2
        return DSE(model_name="checkpoint/dse-phi3-v1", bs=bs)

    else:
        raise ValueError("the model name is not correct!")


def initialize_args():
    '''
    Example: python encode.py BGE --mode vlm_text --encode query,page,layout
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model name, e.g. BGE')
    parser.add_argument('--bs', type=int, default=-1)
    parser.add_argument('--encode_path', type=str, default='encode')
    parser.add_argument('--encode', type=str, default="query,page,layout")
    parser.add_argument('--mode', choices=['vlm_text', 'oct_text', 'image_binary', 'image_hybrid'], default='vlm_text')
    return parser.parse_args()

if __name__ == "__main__":
    # ["BGE", "E5", "GTE", "Contriever", "DPR", "ColBERT", "ColPali", "ColQwen", "DSE-docmatix", "DSE-wikiss"]
    args = initialize_args()
    model, mode, encode, encode_path, bs = args.model, args.mode, args.encode, args.encode_path, args.bs

    retriever = get_retriever(model, bs)

    if "query" in encode:
        # encoding queries
        query_list, query_indices = get_queries("dataset/MMDocIR_annotations.jsonl")
        encoded_query = retriever.embed_queries(query_list)
        print("number of queries to be encoded: ", len(encoded_query))
        with open(f"{encode_path}/encoded_query_{model}_{bs}.pkl", "wb") as f:
            pickle.dump((encoded_query, query_indices), f)
        print("query encoding is done!")

    if "page" in encode:
        # encoding pages
        quote_list, quote_indices = get_pages("dataset/MMDocIR_pages.parquet", mode)
        encoded_quote = retriever.embed_quotes(quote_list)
        print("number of pages to be encoded: ", len(encoded_quote))
        with open(f"{encode_path}/encoded_page_{model}_{bs}.pkl", "wb") as f:
            pickle.dump((encoded_quote, quote_indices), f)
        print("page encoding is done!")

    if "layout" in encode and mode=='image_binary':
        layout_list, layout_indices = get_layouts("dataset/MMDocIR_layouts.parquet", mode)
        print("number of layouts to be encoded: ", len(layout_list))
        # encoding layouts
        encoded_layout = retriever.embed_quotes(layout_list)
        with open(f"{encode_path}/encoded_layout_{model}_{bs}.pkl", "wb") as f:
            pickle.dump((encoded_layout, layout_indices), f)
        print("layout encoding is done!")

    if "layout" in encode and mode=='image_hybrid':
        q_img_list, q_img_indices, q_txt_list, q_txt_indices = get_layouts_hybrid("dataset/MMDocIR_layouts.parquet")
        print("number of layouts to be encoded: ", len(q_img_list)+len(q_txt_list))
        encoded_layout1 = retriever.embed_quotes(q_txt_list, hybrid=True)
        encoded_layout2 = retriever.embed_quotes(q_img_list)

        all_indices = q_txt_indices + q_img_indices
        all_encodings = list(encoded_layout1) + list(encoded_layout2)
        paired = list(zip(all_indices, all_encodings))
        paired_sorted = sorted(paired, key=lambda x: x[0][0])
        sorted_indices, sorted_encodings = zip(*paired_sorted) # Unpack
        sorted_indices = list(sorted_indices)
        sorted_encodings = list(sorted_encodings)
        with open(f"{encode_path}/encoded_layout_{model}_{bs}_hybrid.pkl", "wb") as f:
            pickle.dump((sorted_encodings, sorted_indices), f)
        print("layout encoding is done!")