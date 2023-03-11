device="cuda:2"

for i in 0 1 2; do
    for model_name in nlp-waseda/roberta-large-japanese ku-nlp/deberta-v2-large-japanese; do
        for batch_size in 512 256 128 64; do
            for lr in 5e-5 3e-5 1e-5; do
                poetry run python src/train_sup.py \
                    --dataset_name nu-snli \
                    --model_name $model_name \
                    --batch_size $batch_size \
                    --lr $lr \
                    --use_jumanpp \
                    --device $device

                poetry run python src/train_sup.py \
                    --dataset_name nu-mnli \
                    --model_name $model_name \
                    --batch_size $batch_size \
                    --lr $lr \
                    --use_jumanpp \
                    --device $device

                poetry run python src/train_sup.py \
                    --dataset_name nu-snli+mnli \
                    --model_name $model_name \
                    --batch_size $batch_size \
                    --lr $lr \
                    --use_jumanpp \
                    --device $device
            done
        done
    done
done
