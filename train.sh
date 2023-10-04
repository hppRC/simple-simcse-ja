# sample training script

poetry run python src/train_sup.py \
    --model_name cl-tohoku/bert-large-japanese-v2 \
    --dataset_name jsnli \
    --batch_size 512 \
    --lr 5e-05 \
    --gradient_checkpointing \
    --save_model \
    --save_model_name cl-nagoya/sup-simcse-ja-large \
    --device "cuda:0"

poetry run python src/train_unsup.py \
    --model_name cl-tohoku/bert-large-japanese-v2 \
    --dataset_name wiki40b \
    --batch_size 64 \
    --lr 3e-05 \
    --gradient_checkpointing \
    --save_model \
    --save_model_name cl-nagoya/unsup-simcse-ja-large \
    --device "cuda:1"

poetry run python src/train_sup.py \
    --model_name cl-tohoku/bert-base-japanese-v3 \
    --dataset_name jsnli \
    --batch_size 512 \
    --lr 5e-05 \
    --gradient_checkpointing \
    --save_model \
    --save_model_name cl-nagoya/sup-simcse-ja-base \
    --device "cuda:2"

poetry run python src/train_unsup.py \
    --model_name cl-tohoku/bert-base-japanese-v3 \
    --dataset_name wiki40b \
    --batch_size 64 \
    --lr 5e-05 \
    --gradient_checkpointing \
    --save_model \
    --save_model_name cl-nagoya/unsup-simcse-ja-base \
    --device "cuda:3"
