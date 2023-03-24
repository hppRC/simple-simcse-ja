device="cuda:3"

for model_name in ku-nlp/roberta-large-japanese-char-wwm cl-tohoku/bert-base-japanese-char-v2; do
    for lr in 1e-5 3e-5 5e-5; do
        for batch_size in 64 128 256 512; do
            for dataset_name in nu-snli nu-mnli nu-snli+mnli; do
                poetry run python src/train_sup.py \
                    --dataset_name $dataset_name \
                    --model_name $model_name \
                    --batch_size $batch_size \
                    --lr $lr \
                    --gradient_checkpointing \
                    --device $device
            done
        done
    done
done
