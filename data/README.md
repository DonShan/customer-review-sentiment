# Dataset

## Primary training source

The production model is trained on a **20,000-row subset** of
[`fancyzhx/amazon_polarity`](https://huggingface.co/datasets/fancyzhx/amazon_polarity)
(the current Hugging Face id for Amazon product review polarity).

| Field | Description |
|-------|-------------|
| Source | Hugging Face `fancyzhx/amazon_polarity` (Amazon product reviews) |
| Original size | ~4 million reviews |
| Subset used | 16,000 train + 4,000 test (stratified) |
| Labels | `0` = negative, `1` = positive |
| License | See the dataset card on Hugging Face |

`amazon_polarity` is a binary sentiment corpus built from Amazon product reviews.
It matches the e-commerce / food-delivery use case: short customer comments that
need to be classified quickly at scale.

## Sample file in this repo

[`sample_reviews.csv`](sample_reviews.csv) is a small, hand-written set of
food-delivery style reviews used to demo the UI and API. It is **not** the
training set.

Do not commit the full Hugging Face download. The training script fetches the
subset at runtime and writes model artifacts to `../models/`.
