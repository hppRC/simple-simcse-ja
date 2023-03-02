poetry run python src/datasets/wikipedia.py &
poetry run python src/datasets/wiki40b.py &
poetry run python src/datasets/cc100.py &
poetry run python src/datasets/bccwj.py &

poetry run python src/datasets/nli.py &

poetry run python src/datasets/sts.py &

wait
