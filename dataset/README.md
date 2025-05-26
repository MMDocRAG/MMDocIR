## Evaluation Datasets

 `MMDocIR_pages.parquet` contains 20,395 document page screenshots from 313 documents. The parquet file is formatted as:

| Name           | Type   | Description                      |
| -------------- | ------ | -------------------------------- |
| `doc_name`     | string | Document name                    |
| `domain`       | string | Document's domain or category    |
| `passage_id`   | string | Identifier for a page id         |
| `image_path`   | string | File path to the page screenshot |
| `image_binary` | binary | JPEG image in binary data        |
| `ocr_text`     | string | Text extracted via OCR           |
| `vlm_text`     | string | Text from Vision-Language Model  |



 `MMDocIR_layouts.parquet` contains 170,338 document layouts from 313 documents. The parquet file is formatted as:

| Name         | Type    | Description                         |
|--------------|---------|-------------------------------------|
| `doc_name`   | string  | Document name                       |
| `domain`     | string  | Document's domain or category       |
| `type`       | string  | Layout type                         |
| `layout_id`  | int     | Identifier for the layout id        |
| `page_id`    | int     | Page identifier of this layout    |
| `image_path` | string  | File path to the layout image cropped from page screenshot |
| `image_binary` | binary  | JPEG image in binary data           |
| `text`      | string  | Text content (only applicable to `text` and `equation` layouts) |
| `ocr_text`   | string  | Text extracted via OCR (only applicable to `figure` and `table` layouts) |
| `vlm_text`   | string  | Text from VLM (only applicable to `figure` and `table` layouts) |
| `bbox`       | list[]  | Bounding box coordinates: `[x1,y1,x2,y2]` |
| `page_size`  | list[]  | Page size as `[width, height]`      |







## Training Datasets

coming soon
