device="cuda:2"
model_name="cl-tohoku/bert-base-japanese-v2"

for i in 0 1 2; do
    for batch_size in 32 64 128 256 512; do
        for lr in 1e-5 3e-5 5e-5; do
            poetry run python src/train_unsup.py \
                --dataset_name wiki40b \
                --model_name $model_name \
                --batch_size $batch_size \
                --lr $lr \
                --device $device
        done
    done
done
