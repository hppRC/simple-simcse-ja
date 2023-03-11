device="cuda:2"

for model_name in ku-nlp/deberta-v2-base-japanese nlp-waseda/roberta-base-japanese megagonlabs/roberta-long-japanese; do
    for i in 0 1 2; do
        for batch_size in 64 128 256 512; do
            for lr in 1e-5 3e-5 5e-5; do
                poetry run python src/train_unsup.py \
                    --dataset_name cc100 \
                    --model_name $model_name \
                    --batch_size $batch_size \
                    --lr $lr \
                    --use_jumanpp \
                    --device $device

                poetry run python src/train_unsup.py \
                    --dataset_name bccwj \
                    --model_name $model_name \
                    --batch_size $batch_size \
                    --lr $lr \
                    --use_jumanpp \
                    --device $device
            done
        done
    done
done
