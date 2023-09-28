# Japanese Simple-SimCSE

poetry install
<!-- poetry run pip install --upgrade --force-reinstall --no-deps "apache-beam[gcp]" "multiprocess==0.70.14" -->
<!-- poetry run pip install --upgrade --force-reinstall --no-deps "apache-beam[gcp]" "multiprocess==0.70.14" "dill==0.3.1.1" -->


## Overall
dataset: jsnli, wiki40b

### Supervised SimCSE

| Base models                                                                                                               | JSICK (val) | JSICK (test) | JSTS (train) | JSTS (val) |   Avg.    |
| ------------------------------------------------------------------------------------------------------------------------- | :---------: | :----------: | :----------: | :--------: | :-------: |
| [cl-tohoku/bert-base-japanese-v3](https://huggingface.co/cl-tohoku/bert-base-japanese-v3)                                 |    83.60    |    82.66     |    77.34     |   80.70    |   80.23   |
| [cl-tohoku/bert-base-japanese-v2](https://huggingface.co/cl-tohoku/bert-base-japanese-v2)                                 |    84.20    |    83.39     |    77.03     |   80.70    | **80.37** |
| [cl-tohoku/bert-base-japanese](https://huggingface.co/cl-tohoku/bert-base-japanese)                                       |    83.39    |    82.44     |    75.25     |   78.46    |   78.72   |
| [cl-tohoku/bert-base-japanese-whole-word-masking](https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking) |    83.29    |    82.32     |    75.79     |   79.01    |   79.04   |
| [studio-ousia/luke-japanese-base-lite](https://huggingface.co/studio-ousia/luke-japanese-base-lite)                       |    82.89    |    81.72     |    75.64     |   79.34    |   78.90   |
|                                                                                                                           |             |              |              |            |           |
| [ku-nlp/deberta-v2-base-japanese](https://huggingface.co/ku-nlp/deberta-v2-base-japanese)                                 |    81.90    |    80.78     |    74.71     |   78.39    |   77.96   |
| [nlp-waseda/roberta-base-japanese](https://huggingface.co/nlp-waseda/roberta-base-japanese)                               |    82.94    |    82.00     |    75.65     |   79.63    |   79.09   |
| [megagonlabs/roberta-long-japanese](https://huggingface.co/megagonlabs/roberta-long-japanese)                             |    82.25    |    80.77     |    72.39     |   76.54    |   76.57   |
|                                                                                                                           |             |              |              |            |           |
| [cl-tohoku/bert-base-japanese-char-v3](https://huggingface.co/cl-tohoku/bert-base-japanese-char-v3)                       |    82.57    |    81.35     |    75.75     |   78.62    |   78.57   |
| [cl-tohoku/bert-base-japanese-char-v2](https://huggingface.co/cl-tohoku/bert-base-japanese-char-v2)                       |    83.38    |    81.95     |    74.98     |   78.64    |   78.52   |
| [cl-tohoku/bert-base-japanese-char](https://huggingface.co/cl-tohoku/bert-base-japanese-char)                             |    82.89    |    81.40     |    74.35     |   77.79    |   77.85   |
| [ku-nlp/roberta-base-japanese-char-wwm](https://huggingface.co/ku-nlp/roberta-base-japanese-char-wwm)                     |    82.80    |    80.62     |    74.35     |   78.54    |   77.84   |
|                                                                                                                           |             |              |              |            |           |
| [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)                                       |    83.46    |    82.12     |    73.33     |   76.82    |   77.42   |
| [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)                                                               |    80.29    |    78.42     |    72.54     |   76.02    |   75.66   |
| [studio-ousia/mluke-base-lite](https://huggingface.co/studio-ousia/mluke-base-lite)                                       |    83.48    |    81.96     |    74.97     |   78.47    |   78.47   |

| Large models                                                                                            | JSICK (val) | JSICK (test) | JSTS (train) | JSTS (val) |   Avg.    |
| ------------------------------------------------------------------------------------------------------- | :---------: | :----------: | :----------: | :--------: | :-------: |
| [cl-tohoku/bert-large-japanese-v2](https://huggingface.co/cl-tohoku/bert-large-japanese-v2)             |    83.97    |    82.63     |    79.44     |   82.98    | **81.68** |
| [cl-tohoku/bert-large-japanese](https://huggingface.co/cl-tohoku/bert-large-japanese)                   |    83.70    |    82.54     |    76.49     |   80.09    |   79.71   |
| [studio-ousia/luke-japanese-large-lite](https://huggingface.co/studio-ousia/luke-japanese-large-lite)   |    83.82    |    82.50     |    78.94     |   82.24    |   81.23   |
|                                                                                                         |             |              |              |            |           |
| [nlp-waseda/roberta-large-japanese](https://huggingface.co/nlp-waseda/roberta-large-japanese)           |    84.42    |    83.08     |    79.28     |   82.63    | **81.66** |
| [ku-nlp/deberta-v2-large-japanese](https://huggingface.co/ku-nlp/deberta-v2-large-japanese)             |    79.81    |    79.47     |    77.32     |   80.29    |   79.03   |
|                                                                                                         |             |              |              |            |           |
| [cl-tohoku/bert-large-japanese-char-v2](https://huggingface.co/cl-tohoku/bert-large-japanese-char-v2)   |    83.63    |    82.14     |    77.97     |   80.88    |   80.33   |
| [ku-nlp/roberta-large-japanese-char-wwm](https://huggingface.co/ku-nlp/roberta-large-japanese-char-wwm) |    83.30    |    81.87     |    77.54     |   80.90    |   80.10   |
|                                                                                                         |             |              |              |            |           |
| [xlm-roberta-large](https://huggingface.co/xlm-roberta-large)                                           |    83.59    |    82.04     |    76.63     |   79.91    |   79.53   |
| [studio-ousia/mluke-large-lite](https://huggingface.co/studio-ousia/mluke-large-lite)                   |    84.02    |    82.34     |    77.69     |   80.01    |   80.01   |


### Unsupervised SimCSE

| Base models                                                                                                               | JSICK (val) | JSICK (test) | JSTS (train) | JSTS (val) |   Avg.    |
| ------------------------------------------------------------------------------------------------------------------------- | :---------: | :----------: | :----------: | :--------: | :-------: |
| [cl-tohoku/bert-base-japanese-v3](https://huggingface.co/cl-tohoku/bert-base-japanese-v3)                                 |    79.17    |    78.47     |    74.82     |   78.70    | **77.33** |
| [cl-tohoku/bert-base-japanese-v2](https://huggingface.co/cl-tohoku/bert-base-japanese-v2)                                 |    80.25    |    79.72     |    72.75     |   77.65    |   76.71   |
| [cl-tohoku/bert-base-japanese](https://huggingface.co/cl-tohoku/bert-base-japanese)                                       |    76.94    |    76.90     |    72.29     |   75.92    |   75.04   |
| [cl-tohoku/bert-base-japanese-whole-word-masking](https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking) |    77.52    |    77.37     |    73.23     |   77.14    |   75.91   |
|                                                                                                                           |             |              |              |            |           |
| [studio-ousia/luke-japanese-base-lite](https://huggingface.co/studio-ousia/luke-japanese-base-lite)                       |    81.29    |    80.29     |    72.91     |   78.12    |   77.11   |
| [ku-nlp/deberta-v2-base-japanese](https://huggingface.co/ku-nlp/deberta-v2-base-japanese)                                 |    75.51    |    75.23     |    72.07     |   76.54    |   74.61   |
| [nlp-waseda/roberta-base-japanese](https://huggingface.co/nlp-waseda/roberta-base-japanese)                               |    77.54    |    77.47     |    74.09     |   78.95    |   76.84   |
| [megagonlabs/roberta-long-japanese](https://huggingface.co/megagonlabs/roberta-long-japanese)                             |    74.53    |    73.95     |    63.10     |   68.72    |   68.59   |
|                                                                                                                           |             |              |              |            |           |
| [cl-tohoku/bert-base-japanese-char-v3](https://huggingface.co/cl-tohoku/bert-base-japanese-char-v3)                       |    78.39    |    78.18     |    73.36     |   77.74    |   76.42   |
| [cl-tohoku/bert-base-japanese-char-v2](https://huggingface.co/cl-tohoku/bert-base-japanese-char-v2)                       |    79.29    |    79.00     |    71.36     |   75.60    |   75.32   |
| [cl-tohoku/bert-base-japanese-char](https://huggingface.co/cl-tohoku/bert-base-japanese-char)                             |    77.27    |    76.94     |    69.25     |   73.00    |   73.07   |
| [ku-nlp/roberta-base-japanese-char-wwm](https://huggingface.co/ku-nlp/roberta-base-japanese-char-wwm)                     |    72.21    |    72.21     |    69.73     |   74.69    |   72.21   |
|                                                                                                                           |             |              |              |            |           |
| [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)                                       |    78.45    |    78.23     |    67.60     |   72.36    |   72.73   |
| [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)                                                               |    78.70    |    78.37     |    66.63     |   71.28    |   72.09   |
| [studio-ousia/mluke-base-lite](https://huggingface.co/studio-ousia/mluke-base-lite)                                       |    80.38    |    79.83     |    70.79     |   75.31    |   75.31   |


| Large models                                                                                            | JSICK (val) | JSICK (test) | JSTS (train) | JSTS (val) |   Avg.    |
| ------------------------------------------------------------------------------------------------------- | :---------: | :----------: | :----------: | :--------: | :-------: |
| [cl-tohoku/bert-large-japanese-v2](https://huggingface.co/cl-tohoku/bert-large-japanese-v2)             |    79.54    |    79.14     |    77.18     |   81.00    |   79.11   |
| [cl-tohoku/bert-large-japanese](https://huggingface.co/cl-tohoku/bert-large-japanese)                   |    78.54    |    78.30     |    72.87     |   76.74    |   75.97   |
| [studio-ousia/luke-japanese-large-lite](https://huggingface.co/studio-ousia/luke-japanese-large-lite)   |    79.02    |    78.64     |    75.61     |   79.71    |   77.99   |
|                                                                                                         |             |              |              |            |           |
| [nlp-waseda/roberta-large-japanese](https://huggingface.co/nlp-waseda/roberta-large-japanese)           |    82.94    |    82.56     |    76.04     |   81.28    | **79.96** |
| [ku-nlp/deberta-v2-large-japanese](https://huggingface.co/ku-nlp/deberta-v2-large-japanese)             |    74.60    |    74.95     |    73.49     |   77.34    |   75.26   |
|                                                                                                         |             |              |              |            |           |
| [cl-tohoku/bert-large-japanese-char-v2](https://huggingface.co/cl-tohoku/bert-large-japanese-char-v2)   |    79.07    |    78.73     |    75.68     |   79.10    |   77.83   |
| [ku-nlp/roberta-large-japanese-char-wwm](https://huggingface.co/ku-nlp/roberta-large-japanese-char-wwm) |    76.89    |    76.88     |    72.93     |   77.52    |   75.78   |
|                                                                                                         |             |              |              |            |           |
| [xlm-roberta-large](https://huggingface.co/xlm-roberta-large)                                           |    81.50    |    80.80     |    74.66     |   79.09    |   78.18   |
| [studio-ousia/mluke-large-lite](https://huggingface.co/studio-ousia/mluke-large-lite)                   |    80.38    |    79.44     |    73.11     |   77.66    |   76.74   |

## Dataset

### sup-simcse

| Dataset      | JSICK (val) | JSICK (test) | JSTS (train) | JSTS (val) |   Avg.    |
| ------------ | :---------: | :----------: | :----------: | :--------: | :-------: |
| JSNLI        |    83.97    |    82.63     |    79.44     |   82.98    | **81.68** |
| NU-SNLI      |    83.62    |    82.49     |    79.20     |   82.34    |   81.34   |
| NU-SNLI+MNLI |    82.44    |    81.87     |    79.59     |   82.81    |   81.43   |
| JaNLI        |    81.10    |    80.92     |    73.41     |   77.98    |   77.44   |
| NU-MNLI      |    75.65    |    75.39     |    81.13     |   83.53    |   80.02   |
### unsup-simcse

| Dataset   | JSICK (val) | JSICK (test) | JSTS (train) | JSTS (val) |   Avg.    |
| --------- | :---------: | :----------: | :----------: | :--------: | :-------: |
| Wikipedia |    79.63    |    79.40     |    77.18     |   80.28    |   78.95   |
| BCCWJ     |    79.60    |    79.45     |    76.71     |   80.83    |   79.00   |
| Wiki40b   |    79.54    |    79.14     |    77.18     |   81.00    | **79.11** |
| CC100     |    76.26    |    76.27     |    71.39     |   75.91    |   74.52   |