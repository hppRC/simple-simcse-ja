device="cuda:1"

for batch_size in 1024; do
    for i in 0 1; do
        for model_name in nlp-waseda/roberta-large-japanese ku-nlp/deberta-v2-large-japanese; do
            for lr in 1e-5 3e-5 5e-5; do
                poetry run python src/train_sup.py \
                    --dataset_name nu-snli \
                    --model_name $model_name \
                    --batch_size $batch_size \
                    --lr $lr \
                    --use_jumanpp \
                    --gradient_checkpointing \
                    --device $device

                poetry run python src/train_sup.py \
                    --dataset_name nu-snli+mnli \
                    --model_name $model_name \
                    --batch_size $batch_size \
                    --lr $lr \
                    --use_jumanpp \
                    --gradient_checkpointing \
                    --device $device
            done
        done
    done
done
