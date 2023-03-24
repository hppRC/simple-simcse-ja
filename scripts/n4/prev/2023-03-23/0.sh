device="cuda:0"

for i in 0 1 2; do
    for model_name in nlp-waseda/roberta-large-japanese; do
        for lr in 1e-5 3e-5 5e-5; do
            for batch_size in 512; do
                poetry run python src/train_unsup.py \
                    --dataset_name wiki40b \
                    --model_name $model_name \
                    --batch_size $batch_size \
                    --lr $lr \
                    --use_jumanpp \
                    --gradient_checkpointing \
                    --device $device
            done
        done
    done

    for model_name in studio-ousia/luke-japanese-large-lite studio-ousia/luke-japanese-base-lite; do
        for lr in 1e-5 3e-5 5e-5; do
            for batch_size in 512; do
                poetry run python src/train_unsup.py \
                    --dataset_name wiki40b \
                    --model_name $model_name \
                    --batch_size $batch_size \
                    --lr $lr \
                    --gradient_checkpointing \
                    --device $device
            done
        done
    done
done
