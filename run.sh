models=(
    studio-ousia/luke-japanese-base-lite
    studio-ousia/luke-japanese-large-lite
    cl-tohoku/bert-base-japanese-v2
    cl-tohoku/bert-base-japanese-whole-word-masking
    cl-tohoku/bert-base-japanese
    cl-tohoku/bert-large-japanese
    studio-ousia/mluke-base-lite
    studio-ousia/mluke-large-lite
    cl-tohoku/roberta-base-japanese
    cl-tohoku/bert-base-japanese-char-whole-word-masking
    cl-tohoku/bert-base-japanese-char-v2
    cl-tohoku/bert-base-japanese-char
    cl-tohoku/bert-large-japanese-char
    # nlp-waseda/roberta-base-japanese-with-auto-jumanpp
    # nlp-waseda/roberta-large-japanese-with-auto-jumanpp
    rinna/japanese-roberta-base
    Cinnamon/electra-small-japanese-generator
    Cinnamon/electra-small-japanese-discriminator
    ptaszynski/yacis-electra-small-japanese
    bert-base-multilingual-cased
    ganchengguang/RoBERTa-base-janpanese
    megagonlabs/roberta-long-japanese
    megagonlabs/electra-base-japanese-discriminator
    bandainamco-mirai/distilbert-base-japanese
    izumi-lab/bert-small-japanese-fin
    izumi-lab/bert-small-japanese
)

# for model_name in "${models[@]}"; do
#     poetry run python train.py \
#         --model_name $model_name \
#         --device "cuda:1"
# done

for model_name in "${models[@]}"; do
    for dataset_name in jnli janli jsnli; do
        poetry run python train.py \
            --method sup-simcse \
            --epochs 5 \
            --batch_size 64 \
            --dataset_name $dataset_name \
            --model_name $model_name \
            --device "cuda:0"
        poetry run python train.py \
            --method sup-simcse \
            --epochs 5 \
            --batch_size 128 \
            --dataset_name $dataset_name \
            --model_name $model_name \
            --device "cuda:0"
        poetry run python train.py \
            --method sup-simcse \
            --epochs 5 \
            --batch_size 256 \
            --dataset_name $dataset_name \
            --model_name $model_name \
            --device "cuda:0"
    done
done
