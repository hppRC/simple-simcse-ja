device="cuda:3"

for i in 0 1 2; do
    for model_name in microsoft/mdeberta-v3-base studio-ousia/mluke-base-lite; do
        for lr in 1e-5 3e-5 5e-5; do
            for batch_size in 512; do
                poetry run python src/train_sup.py \
                    --dataset_name jsnli+nu-snli \
                    --model_name $model_name \
                    --batch_size $batch_size \
                    --lr $lr \
                    --gradient_checkpointing \
                    --device $device

                poetry run python src/train_sup.py \
                    --dataset_name jsnli \
                    --model_name $model_name \
                    --batch_size $batch_size \
                    --lr $lr \
                    --gradient_checkpointing \
                    --device $device
            done
            for batch_size in 256 128 64; do
                poetry run python src/train_sup.py \
                    --dataset_name jsnli+nu-snli \
                    --model_name $model_name \
                    --batch_size $batch_size \
                    --lr $lr \
                    --device $device
            done
        done
    done
done
