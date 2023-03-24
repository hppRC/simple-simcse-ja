device="cuda:3"

for i in 0 1 2; do
    for model_name in bert-base-multilingual-cased xlm-roberta-base microsoft/mdeberta-v3-base studio-ousia/mluke-base-lite; do
        for batch_size in 512 256 128 64; do
            for lr in 5e-5 3e-5 1e-5; do
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
