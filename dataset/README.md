# Evaluation Datasets



## Dataset Overview

**MMDocIR** evaluation set includes 313 long documents averaging 65.1 pages, categorized into ten main domains: research reports, administration&industry, tutorials&workshops, academic papers, brochures, financial reports, guidebooks, government documents, laws, and news articles. Different domains feature distinct distributions of multi-modal information. Overall, the modality distribution is: Text (60.4%), Image (18.8%), Table (16.7%), and other modalities (4.1%).

**MMDocIR** evluation set encompasses 1,658 questions, 2,107 page labels, and 2,638 layout labels. The modalities required to answer these questions distribute across four categories: Text (44.7%), Image (21.7%), Table (37.4%), and Layout/Meta (11.5%). The ``Layout/Meta'' category encompasses questions related to layout information and meta-data statistics. Notably, the dataset poses several challenges: 254 questions necessitate cross-modal understanding, 313 questions demand evidence across multiple pages, and 637 questions require reasoning based on multiple layouts. These complexities highlight the need for advanced multi-modal reasoning and contextual understanding.



## Dataset Format

[`MMDocIR_annotations.jsonl`](https://github.com/MMDocRAG/MMDocIR/blob/main/dataset/MMDocIR_annotations.jsonl) contains 313 json lines, each for the annotations corresponding to a long document.

| Name              | Type   | Description                                                  |
| ----------------- | ------ | ------------------------------------------------------------ |
| `doc_name`        | string | document name                                                |
| `domain`          | string | document's domain or category                                |
| `page_indices`    | list[] | start/end row index (not passage id) in [`MMDocIR_pages.parquet`](https://huggingface.co/datasets/MMDocIR/MMDocIR_Evaluation_Dataset/blob/main/MMDocIR_pages.parquet) |
| `layout_indinces` | list[] | start/end row index (not layout id) in [`MMDocIR_layouts.parquet`](https://huggingface.co/datasets/MMDocIR/MMDocIR_Evaluation_Dataset/blob/main/MMDocIR_layouts.parquet) |
| `questions`       | list[] | list of QA items with `page_id`, `type`, `layout_mapping`    |

Each QA item consists of :
| Name             | Type   | Description                                            |
| ---------------- | ------ | ------------------------------------------------------ |
| `Q`              | string | questions                                              |
| `A`              | string | answer                                                 |
| `page_id`        | list[] | the list of page ids for ground truth evidence         |
| `type`           | string | the modality type of the question                      |
| `layout_mapping` | list[] | list of layout labels with `page`, `page_size`, `bbox` |



[`MMDocIR_pages.parquet`](https://huggingface.co/datasets/MMDocIR/MMDocIR_Evaluation_Dataset/blob/main/MMDocIR_pages.parquet) contains 20,395 document page screenshots from 313 documents. The parquet file is formatted as:

| Name           | Type   | Description                      |
| -------------- | ------ | -------------------------------- |
| `doc_name`     | string | document name                    |
| `domain`       | string | document's domain or category    |
| `passage_id`   | string | identifier for a page id         |
| `image_path`   | string | file path to the page screenshot |
| `image_binary` | binary | JPEG image in binary data        |
| `ocr_text`     | string | text extracted via OCR           |
| `vlm_text`     | string | text from Vision-Language Model  |



[`MMDocIR_layouts.parquet`](https://huggingface.co/datasets/MMDocIR/MMDocIR_Evaluation_Dataset/blob/main/MMDocIR_layouts.parquet) contains 170,338 document layouts from 313 documents. The parquet file is formatted as:

| Name           | Type   | Description                                                  |
| -------------- | ------ | ------------------------------------------------------------ |
| `doc_name`     | string | document name                                                |
| `domain`       | string | document's domain or category                                |
| `type`         | string | layout type                                                  |
| `layout_id`    | int    | identifier for the layout id                                 |
| `page_id`      | int    | page identifier of this layout                               |
| `image_path`   | string | file path to the layout image cropped from page screenshot   |
| `image_binary` | binary | JPEG image in binary data                                    |
| `text`         | string | text content (only applicable to `text` and `equation` layouts) |
| `ocr_text`     | string | text extracted via OCR (only applicable to `figure` and `table` layouts) |
| `vlm_text`     | string | text from VLM (only applicable to `figure` and `table` layouts) |
| `bbox`         | list[] | bounding box coordinates: `[x1,y1,x2,y2]`                    |
| `page_size`    | list[] | page size as `[width, height]`                               |







# Training Datasets

coming soon
