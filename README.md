# Japanese Simple-SimCSE

poetry install
<!-- poetry run pip install --upgrade --force-reinstall --no-deps "apache-beam[gcp]" "multiprocess==0.70.14" -->
poetry run pip install --upgrade --force-reinstall --no-deps "apache-beam[gcp]" "multiprocess==0.70.14" "dill==0.3.1.1"



| base models                                                                                                               | batch size |  lr   | JSICK (val) | JSICK (test) | JSTS (train) | JSTS (val) |
| ------------------------------------------------------------------------------------------------------------------------- | :--------: | :---: | :---------: | :----------: | :----------: | :--------: |
| [cl-tohoku/bert-base-japanese-v2](https://huggingface.co/cl-tohoku/bert-base-japanese-v2)                                 |            |       |             |              |              |            |
| [cl-tohoku/bert-base-japanese-char-v2](https://huggingface.co/cl-tohoku/bert-base-japanese-char-v2)                       |            |       |             |              |              |            |
| [cl-tohoku/bert-base-japanese](https://huggingface.co/cl-tohoku/bert-base-japanese)                                       |            |       |             |              |              |            |
| [cl-tohoku/bert-base-japanese-whole-word-masking](https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking) |            |       |             |              |              |            |
| [cl-tohoku/bert-base-japanese-char](https://huggingface.co/cl-tohoku/bert-base-japanese-char)                             |            |       |             |              |              |            |
| [ku-nlp/roberta-base-japanese-char-wwm](https://huggingface.co/ku-nlp/roberta-base-japanese-char-wwm)                     |            |       |             |              |              |            |
| [studio-ousia/luke-japanese-base-lite](https://huggingface.co/studio-ousia/luke-japanese-base-lite)                       |            |       |             |              |              |            |
|                                                                                                                           |            |       |             |              |              |            |
| [ku-nlp/deberta-v2-base-japanese](https://huggingface.co/ku-nlp/deberta-v2-base-japanese)                                 |            |       |             |              |              |            |
| [nlp-waseda/roberta-base-japanese](https://huggingface.co/nlp-waseda/roberta-base-japanese)                               |            |       |             |              |              |            |
| [megagonlabs/roberta-long-japanese](https://huggingface.co/megagonlabs/roberta-long-japanese)                             |            |       |             |              |              |            |
|                                                                                                                           |            |       |             |              |              |            |
| [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)                                       |            |       |             |              |              |            |
| [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)                                                               |            |       |             |              |              |            |
| [microsoft/mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base)                                           |            |       |             |              |              |            |
| [studio-ousia/mluke-base-lite](https://huggingface.co/studio-ousia/mluke-base-lite)                                       |            |       |             |              |              |            |



| large models                                                                                            | batch size |  lr   | JSICK (val) | JSICK (test) | JSTS (train) | JSTS (val) |
| ------------------------------------------------------------------------------------------------------- | :--------: | :---: | :---------: | :----------: | :----------: | :--------: |
| [cl-tohoku/bert-large-japanese](https://huggingface.co/cl-tohoku/bert-large-japanese)                   |            |       |             |              |              |            |
| [ku-nlp/roberta-large-japanese-char-wwm](https://huggingface.co/ku-nlp/roberta-large-japanese-char-wwm) |            |       |             |              |              |            |
| [studio-ousia/luke-japanese-large-lite](https://huggingface.co/studio-ousia/luke-japanese-large-lite)   |            |       |             |              |              |            |
|                                                                                                         |            |       |             |              |              |            |
| [nlp-waseda/roberta-large-japanese](https://huggingface.co/nlp-waseda/roberta-large-japanese)           |            |       |             |              |              |            |
| [ku-nlp/deberta-v2-large-japanese](https://huggingface.co/ku-nlp/deberta-v2-large-japanese)             |            |       |             |              |              |            |
|                                                                                                         |            |       |             |              |              |            |
| [xlm-roberta-large](https://huggingface.co/xlm-roberta-large)                                           |            |       |             |              |              |            |
| [studio-ousia/mluke-large-lite](https://huggingface.co/studio-ousia/mluke-large-lite)                   |            |       |             |              |              |            |
