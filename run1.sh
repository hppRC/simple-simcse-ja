nohup poetry run python src/optimize.py --mlp_type simcse --dtype fp16 --device "cuda:0" --model_name "cl-tohoku/bert-base-japanese-v2" --timeout 43200 >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --mlp_type simcse-post-bn --dtype fp16 --device "cuda:1" --model_name "cl-tohoku/bert-base-japanese-v2" --timeout 43200 >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --mlp_type simcse --dtype fp16 --device "cuda:2" --model_name "studio-ousia/luke-japanese-base-lite" --timeout 43200 >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --mlp_type simcse-post-bn --dtype fp16 --device "cuda:3" --model_name "studio-ousia/luke-japanese-base-lite" --timeout 43200 >/dev/null 2>&1 &

nohup poetry run python src/optimize.py --device "cuda:1" --model_name "studio-ousia/luke-japanese-large-lite" --timeout 32400 --mlp_type bn >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --device "cuda:2" --model_name "studio-ousia/luke-japanese-large-lite" --timeout 32400 --mlp_type simcse >/dev/null 2>&1 &
nohup poetry run python src/optimize.py --device "cuda:3" --model_name "studio-ousia/luke-japanese-large-lite" --timeout 32400 --mlp_type simcse-post-bn >/dev/null 2>&1 &
