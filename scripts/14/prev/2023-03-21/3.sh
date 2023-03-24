device="cuda:3"

for i in 0 1 2; do
    for model_name in studio-ousia/mluke-base-lite bert-base-multilingual-cased xlm-roberta-base microsoft/mdeberta-v3-base; do
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
done
