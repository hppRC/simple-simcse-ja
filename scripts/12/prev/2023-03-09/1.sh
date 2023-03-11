device="cuda:1"

for model_name in studio-ousia/luke-japanese-large-lite ku-nlp/deberta-v2-large-japanese; do
    for batch_size in 64 128 256 512; do
        for lr in 1e-5 3e-5 5e-5; do
            poetry run python src/train_sup.py \
                --dataset_name nu-snli \
                --model_name $model_name \
                --batch_size $batch_size \
                --lr $lr \
                --device $device

            poetry run python src/train_sup.py \
                --dataset_name nu-mnli \
                --model_name $model_name \
                --batch_size $batch_size \
                --lr $lr \
                --device $device

            poetry run python src/train_sup.py \
                --dataset_name nu-snli+mnli \
                --model_name $model_name \
                --batch_size $batch_size \
                --lr $lr \
                --device $device
        done
    done
done
