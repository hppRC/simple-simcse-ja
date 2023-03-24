device="cuda:2"

for i in 0 1 2; do
    for model_name in cl-tohoku/bert-large-japanese studio-ousia/luke-japanese-large-lite; do
        for lr in 1e-5 3e-5 5e-5; do
            for batch_size in 256 512; do
                for dataset_name in wikipedia bccwj cc100; do
                    poetry run python src/train_unsup.py \
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
done
