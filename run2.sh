poetry run python src/train_sup.py \
    --device "cuda:0" \
    --model_name "cl-tohoku/bert-base-japanese-v2" \
    --dataset_dir "./datasets/nli" \
    --dataset_name "jsnli"

poetry run python src/train_sup.py \
    --device "cuda:0" \
    --model_name "cl-tohoku/bert-base-japanese-v2" \
    --dataset_dir "./datasets/nli-translated" \
    --dataset_name "deepl/snli+mnli"

poetry run python src/train_sup.py \
    --device "cuda:0" \
    --model_name "cl-tohoku/bert-base-japanese-v2" \
    --dataset_dir "./datasets/nli" \
    --dataset_name "jsnli+deepl_snli"

poetry run python src/train_sup.py \
    --device "cuda:0" \
    --model_name "cl-tohoku/bert-base-japanese-v2" \
    --dataset_dir "./datasets/nli" \
    --dataset_name "jsnli+deepl_mnli"

poetry run python src/train_sup.py \
    --device "cuda:0" \
    --model_name "cl-tohoku/bert-base-japanese-v2" \
    --dataset_dir "./datasets/nli" \
    --dataset_name "jsnli+deepl_snli+deepl_mnli"

poetry run python src/train_sup.py \
    --device "cuda:0" \
    --model_name "cl-tohoku/bert-base-japanese-v2" \
    --dataset_dir "./datasets/nli" \
    --dataset_name "jsnli-small"
