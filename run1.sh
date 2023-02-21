nohup poetry run python src/optimize.py --mlp_type simcse --dtype fp16 --device "cuda:0" --model_name "cl-tohoku/bert-base-japanese-v2" --timeout 43200 >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --mlp_type simcse-post-bn --dtype fp16 --device "cuda:1" --model_name "cl-tohoku/bert-base-japanese-v2" --timeout 43200 >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --mlp_type simcse --dtype fp16 --device "cuda:2" --model_name "studio-ousia/luke-japanese-base-lite" --timeout 43200 >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --mlp_type simcse-post-bn --dtype fp16 --device "cuda:3" --model_name "studio-ousia/luke-japanese-base-lite" --timeout 43200 >/dev/null 2>&1 &

nohup poetry run python src/optimize.py --device "cuda:1" --model_name "studio-ousia/luke-japanese-large-lite" --timeout 32400 --mlp_type bn >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --device "cuda:2" --model_name "studio-ousia/luke-japanese-large-lite" --timeout 32400 --mlp_type simcse >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --device "cuda:3" --model_name "studio-ousia/luke-japanese-large-lite" --timeout 32400 --mlp_type simcse-post-bn >/dev/null 2>&1 &

poetry run python src/optimize.py --device "cuda:1" --model_name "studio-ousia/luke-japanese-base-lite" --timeout 3600 --mlp_type simcse-taper

nohup poetry run python src/optimize.py --device "cuda:0" --dtype fp16 --model_name "studio-ousia/luke-japanese-base-lite" --max_batch_size 256 --timeout 43200 --mlp_type simcse-post-bn --dataset_name "jsnli" >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --device "cuda:1" --dtype fp16 --model_name "studio-ousia/luke-japanese-base-lite" --max_batch_size 256 --timeout 43200 --mlp_type simcse-post-bn --dataset_name "jsnli" >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --device "cuda:2" --dtype fp16 --model_name "studio-ousia/luke-japanese-base-lite" --max_batch_size 256 --timeout 43200 --mlp_type simcse --dataset_name "jsnli" >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --device "cuda:3" --dtype fp16 --model_name "studio-ousia/luke-japanese-base-lite" --max_batch_size 256 --timeout 43200 --mlp_type simcse --dataset_name "jsnli" >/dev/null 2>&1 &

nohup poetry run python src/optimize.py --device "cuda:2" --model_name "studio-ousia/luke-japanese-large-lite" --timeout 60000 --mlp_type simcse --dataset_name "jsnli+wiki40b" >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --device "cuda:3" --model_name "studio-ousia/luke-japanese-large-lite" --timeout 60000 --mlp_type simcse-post-bn --dataset_name "jsnli+wiki40b" >/dev/null 2>&1 &

nohup poetry run python src/optimize.py --device "cuda:1" --model_name "studio-ousia/luke-japanese-base-lite" --timeout 60000 --mlp_type simcse --dataset_name "jsnli+wiki40b" >/dev/null 2>&1 &
poetry run python src/optimize.py --device "cuda:1" --model_name "studio-ousia/luke-japanese-base-lite" --n_trials 10 --mlp_type simcse --dataset_name "jsnli+wiki40b-010"

nohup poetry run python src/optimize.py --device "cuda:1" --model_name "studio-ousia/luke-japanese-base-lite" --timeout 50400 --mlp_type simcse --dataset_name "jsnli+wiki40b-010" >/dev/null 2>&1 &

nohup poetry run python src/optimize.py --device "cuda:0" --model_name "cl-tohoku/bert-base-japanese-v2" --timeout 64800 --mlp_type simcse-post-bn --dataset_name "jsnli" >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --device "cuda:2" --model_name "cl-tohoku/bert-base-japanese-v2" --timeout 64800 --mlp_type simcse --dataset_name "jsnli" >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --device "cuda:1" --model_name "cl-tohoku/bert-base-japanese-v2" --timeout 64800 --mlp_type simcse-post-bn --dataset_name "jsnli" >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --device "cuda:3" --model_name "cl-tohoku/bert-base-japanese-v2" --timeout 64800 --mlp_type simcse --dataset_name "jsnli" >/dev/null 2>&1 &

nohup poetry run python src/optimize.py --device "cuda:0" --model_name "studio-ousia/luke-japanese-base-lite" --timeout 3600 --mlp_type simcse-post-bn --dataset_name "jsnli" >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --device "cuda:1" --model_name "studio-ousia/luke-japanese-base-lite" --timeout 3600 --mlp_type simcse --dataset_name "jsnli" >/dev/null 2>&1 &

nohup poetry run python src/optimize.py --device "cuda:2" --model_name "studio-ousia/luke-japanese-base-lite" --timeout 43200 --mlp_type simcse-post-bn --dataset_name "jsnli" >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --device "cuda:3" --model_name "studio-ousia/luke-japanese-base-lite" --timeout 43200 --mlp_type simcse --dataset_name "jsnli" >/dev/null 2>&1 &

nohup poetry run python src/optimize.py --device "cuda:0" --model_name "studio-ousia/luke-japanese-large-lite" --timeout 64800 --mlp_type simcse-post-bn --dataset_name "jsnli" >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --device "cuda:1" --model_name "studio-ousia/luke-japanese-large-lite" --timeout 64800 --mlp_type simcse --dataset_name "jsnli" >/dev/null 2>&1 &

nohup poetry run python src/optimize.py --device "cuda:0" --model_name "studio-ousia/luke-japanese-base-lite" --timeout 57600 --mlp_type simcse --dataset_dir "./datasets/nli-translated" --dataset_name "nllb/snli+mnli" >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --device "cuda:1" --model_name "studio-ousia/luke-japanese-base-lite" --timeout 57600 --mlp_type simcse --dataset_dir "./datasets/nli-translated" --dataset_name "nllb/snli+mnli" >/dev/null 2>&1 &

poetry run python src/train_sup.py --device "cuda:0" --model_name "studio-ousia/luke-japanese-base-lite" --dataset_dir "./datasets/nli-translated" --dataset_name "deepl/snli+mnli"
poetry run python src/train_sup.py --device "cuda:0" --model_name "xlm-roberta-base" --dataset_dir "./datasets/nli-translated" --dataset_name "deepl/snli+mnli"
