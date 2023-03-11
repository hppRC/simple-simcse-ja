device="cuda:3"
# model_name="xlm-roberta-base"
model_name="nlp-waseda/roberta-large-japanese"
# model_name="ku-nlp/deberta-v2-large-japanese"

# for i in 0 1 2; do
#     for batch_size in 64; do
#         for lr in 1e-5 3e-5 5e-5; do
#             poetry run python src/train_unsup.py \
#                 --dataset_name wiki40b \
#                 --model_name $model_name \
#                 --batch_size $batch_size \
#                 --lr $lr \
#                 --use_jumanpp \
#                 --device $device
#         done
#     done
# done

for i in 0 1 2; do
    # for batch_size in 32 64 128 256 512; do
    for batch_size in 128; do
        for lr in 1e-5 3e-5 5e-5; do
            poetry run python src/train_sup.py \
                --dataset_name jsnli \
                --model_name $model_name \
                --batch_size $batch_size \
                --lr $lr \
                --use_jumanpp \
                --device $device
        done
    done
done
