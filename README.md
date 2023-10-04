# Japanese Simple-SimCSE

文埋め込みは自然言語文の蜜ベクトル表現であり、類似文検索や質問応答、最近では検索補助付き生成(Retrieval Augmented Generation: RAG)に盛んに利用されています。

文埋め込みを構成する方法には様々な種類がありますが、近年では事前学習済み言語モデルに対して対照学習(Contrastive Learning)によるfine-tuningを施す手法が高い性能を示しています。
その中でも代表的な手法が[SimCSE (**Sim**ple **C**ontrastive **S**entence **E**mbedding)](https://aclanthology.org/2021.emnlp-main.552)です。

SimCSEには教師なし・教師ありの二つの設定があります。
教師なし設定では、事前学習済み言語モデル中に存在するDropoutモジュールをデータ拡張の手段とみなして「モデルに同じ文を2回入れて、同じ文同士を正例とする」ことで対照学習を行います。
教師あり設定では、自然言語推論(Natural Language Inference: NLI)データセットを利用して「同じ意味の文同士を正例とする」ことで対照学習を行います。

2021年に発表されたSimCSEはもはや文埋め込みのデファクトスタンダードと言えるほど広く高い性能を示しており、多くの派生研究が存在します。
しかし、多くの研究は英語のみで評価を行っており、日本語の文埋め込みモデルはそこまで多くなく、特に日本語文埋め込みを網羅的に評価した研究は存在しません。

そこで、SimCSEをベースに多様な事前学習済み言語モデル・訓練データセット・ハイパーパラメータで訓練を行い、日本語文埋め込みの網羅的な評価を行いました。
具体的には、教師あり設定で7200回、教師なし設定で5760回の実験を実施しました。

本リポジトリではその結果と実験に用いたスクリプトを公開します。
また、実験の結果選定された設定でfine-tuningを行った、事前学習済みの日本語文埋め込みモデルを4つ公開し、また、それらの評価結果も示します。

| Model                                                                                     | Dataset | JSICK (test) | JSTS (val) |
| ----------------------------------------------------------------------------------------- | :------ | :----------: | :--------: |
| [cl-nagoya/sup-simcse-ja-large](https://huggingface.co/cl-nagoya/sup-simcse-ja-large)     | JSNLI   |    83.05     |   83.07    |
| [cl-nagoya/sup-simcse-ja-base](https://huggingface.co/cl-nagoya/sup-simcse-ja-base)       | JSNLI   |    82.75     |   80.86    |
| [cl-nagoya/unsup-simcse-ja-large](https://huggingface.co/cl-nagoya/unsup-simcse-ja-large) | Wiki40b |    79.62     |   81.40    |
| [cl-nagoya/unsup-simcse-ja-base](https://huggingface.co/cl-nagoya/unsup-simcse-ja-base)   | Wiki40b |    79.01     |   78.95    |

これらのモデルは、文埋め込みモデルのライブラリ[Sentence Transformers](https://github.com/UKPLab/sentence-transformers)から簡単に利用できるようにしてあります。
実際の使用例は以下のとおりです。

```python
# サンプルコード
from sentence_transformers import SentenceTransformer
sentences = ["こんにちは、世界！", "文埋め込み最高！文埋め込み最高と叫びなさい", "極度乾燥しなさい"]

model = SentenceTransformer("cl-nagoya/sup-simcse-ja-base")
embeddings = model.encode(sentences)
```

以降の節では、今回行った実験の詳細な設定と、評価結果について記述します。


## 実験設定

本実験の目的は、日本語文埋め込みモデルの網羅的な評価を行い、よいベースラインとなる日本語文埋め込みモデルを構築することです。
そこで今回は、4つの設定について、それぞれの組み合わせで実験を行いました。
具体的には、24種類のモデル、5+4種類のデータセット、4つのバッチサイズ、3つの学習率の組み合わせについて実験を行いました。

より具体的には、以下の設定からそれぞれ実験を行いました。

- モデル: 24種類 (後述)
- データセット
  - 教師あり (5種類)
    - JSNLI
    - JaNLI
    - NU-SNLI
    - NU-MNLI
    - NU-SNLI+MNLI
  - 教師なし (4種類)
    - Wiki40B: Wikipediaに対してクリーニングを行ったデータセット。
    - [BCCWJ](https://clrd.ninjal.ac.jp/bccwj/): 国語研が提供している現代日本語書き言葉均衡コーパス。
    - Wikipedia: 日本語Wikipediaからランダムにサンプリングしてきた文。
    - CC100: 大規模なWebコーパスをある程度フィルタリングしたもの。
- バッチサイズ: 64, 128, 256, 512
- 学習率: 1e-5, 3e-5, 5e-5

学習スクリプトについては、SimCSEの公式実装を参考に、より簡略化して実装した[Simple-SimCSE](https://github.com/hppRC/simple-simcse)をベースに実装しました。
また、温度パラメータについては、事前実験の結果を元に、ハイパーパラメータ探索の候補から外し、SimCSEのデフォルト値の0.05を用いました。

異なる乱数シード値で5回ずつ実験を行い、その平均を評価スコアとしました。
結果として、教師あり設定では7200回、教師なし設定では5760回の実験を実施しました。
各実験では、一定の事例数ごとに開発セットでの評価を行い、最良のcheckpointを各実験の評価時に用いました。

また、各モデル・データセットごとに、各バッチサイズ・学習率の組み合わせについて、開発セットでのスピアマンの順位相関係数が最も高いハイパラを最終的な評価に用いました。
つまり、5回の実験のスコアを平均した結果、モデルごとに最もよいハイパーパラメータを選んで最終的な評価に用いました。

上記データセットのうち、NU-SNLI, NU-MNLI, NU-SNLI+MNLIは、それぞれ、Stanford NLI (SNLI)データセットとMulti-Genre NLI (MNLI)データセットをChatGPT (gpt-3.5-turbo)を用いて英日翻訳したデータから構成される独自のNLIデータセットです。
本来であればこれらをまとめてNU-NLIデータセット(仮称)として公開したかったのですが、ライセンスの問題から現在は公開を見送っている状況です。
ご了承ください。

## 評価設定

評価タスクとして文類似度(Semantic Textual Similarity)タスクを用いました。
評価データセットとしてJSICK ([GitHub](https://github.com/verypluming/JSICK), [HuggingFace]((https://huggingface.co/datasets/hpprc/jsick))と[JSTS](https://huggingface.co/datasets/shunk031/JGLUE)を用いました。

このうち、JSICKの開発(val)セットを文埋め込みモデル訓練中の開発セットとして用い、JSICKのテスト(test)セットとJSTSの訓練(train)・開発(val)セットを最終的な評価に用いました。
つまり、ハイパーパラメータ探索に利用した評価セットは、開発セットとして用いたJSICKの開発セットになります。
また、最終的な評価も5回の実験の平均をとっています。
SimCSEは乱数シード値やハイパラによって性能がブレやすい手法なので、複数回の実験の平均をとることで出来るだけブレを抑えています。

## 評価実験


各モデル・データセットごとに最良のハイパラを選んだ際の評価結果を、教師あり・教師なしそれぞれ`results/sup-simcse/best.csv`および`results/unsup-simcse/best.csv`に格納してあります。
また、すべてのデータセット・すべてのモデル・すべてのハイパーパラメータでの平均スコアは本リポジトリの`results/sup-simcse/all.csv`および`results/unsup-simcse/all.csv`に格納されています。
お好みのモデルで評価をしたい場合に、どのデータセットやハイパラを選べばいいかの参考にぜひお使いください。

以下の節では、教師あり・教師なしのそれぞれについて、実験の結果を示します。

### Supervised SimCSE

まず、データセットとしてJSNLIを用いた場合の、教師あり設定でのモデルごとの結果を示します。
以下の表は比較的小さい(110M程度)の事前学習済み言語モデルを用いた際の結果です。

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

表から、東北大BERTが比較的高い性能を示していることがわかり、早稲田大・京大のRoBERTaが近い性能を示していることがわかります。
また、文字レベルのモデルや、多言語モデルの性能は、比較的低めの性能になっています。

次に、比較的大きなモデルでの評価結果を以下に示します。

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

表から、やはり東北大BERTの性能が高いことがわかります。
また、Studio Ousiaの日本語LUKEも高い性能を示しました。

### Unsupervised SimCSE

次に、データセットとしてWiki40Bを用いた場合の、教師なし設定でのモデルごとの結果を示します。

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

表から、教師なし設定においても、東北大BERT(の特にv3)が高い性能を示したことがわかります。


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

表から、教師なし設定においても、東北大BERT(の特にv2)が高い性能を示しました。

### Model Ranking

次に、データセットに対するモデルの頑健性(データセットが変わってもちゃんと性能が出せるか)を調べるために、データセットごとにモデルの順位をつけ、その平均を計算しました。
その結果が以下の表です。

| Model                                           | Supervised | Unsupervised |
| ----------------------------------------------- | ---------: | -----------: |
| nlp-waseda/roberta-large-japanese               |       1.20 |         1.00 |
| cl-tohoku/bert-large-japanese-v2                |       2.20 |         4.25 |
| studio-ousia/luke-japanese-large-lite           |       2.80 |         3.50 |
| cl-tohoku/bert-base-japanese-v3                 |       6.40 |         4.50 |
| studio-ousia/mluke-large-lite                   |       6.60 |         8.00 |
| cl-tohoku/bert-base-japanese-v2                 |       6.80 |        10.25 |
| cl-tohoku/bert-large-japanese-char-v2           |       7.00 |         5.00 |
| ku-nlp/roberta-large-japanese-char-wwm          |       8.00 |        15.25 |
| cl-tohoku/bert-large-japanese                   |       9.00 |        10.50 |
| studio-ousia/luke-japanese-base-lite            |      10.00 |         7.00 |
| xlm-roberta-large                               |      11.80 |         5.00 |
| nlp-waseda/roberta-base-japanese                |      12.80 |         8.50 |
| ku-nlp/deberta-v2-large-japanese                |      13.20 |        13.25 |
| cl-tohoku/bert-base-japanese-whole-word-masking |      13.80 |        16.75 |
| studio-ousia/mluke-base-lite                    |      15.00 |        17.00 |
| ku-nlp/deberta-v2-base-japanese                 |      15.40 |        15.75 |
| cl-tohoku/bert-base-japanese                    |      16.40 |        18.00 |
| ku-nlp/roberta-base-japanese-char-wwm           |      16.60 |        22.25 |
| cl-tohoku/bert-base-japanese-char-v3            |      17.40 |        12.75 |
| cl-tohoku/bert-base-japanese-char-v2            |      19.60 |        15.25 |
| bert-base-multilingual-cased                    |      20.60 |        20.75 |
| cl-tohoku/bert-base-japanese-char               |      22.00 |        19.50 |
| xlm-roberta-base                                |      22.60 |        22.00 |
| megagonlabs/roberta-long-japanese               |      22.80 |        24.00 |

表から、早稲田大RoBERTaが一貫して高い性能を示していることがわかりました。
早稲田大RoBERTaはJuman++による事前の分かち書きが必要ですが、概して高い性能を発揮しているので、検討する価値がある強いモデルだと思われます。


### Dataset Ranking

次に、データセットごとに分析を行った結果を示します。

#### Supervised SimCSE

以下の表は、教師あり設定において、モデルを`cl-tohoku/bert-large-japanese-v2`に固定した際の、データセットごとの結果です。

| Dataset      | JSICK (val) | JSICK (test) | JSTS (train) | JSTS (val) |   Avg.    |
| ------------ | :---------: | :----------: | :----------: | :--------: | :-------: |
| JSNLI        |    83.97    |    82.63     |    79.44     |   82.98    | **81.68** |
| NU-SNLI      |    83.62    |    82.49     |    79.20     |   82.34    |   81.34   |
| NU-SNLI+MNLI |    82.44    |    81.87     |    79.59     |   82.81    |   81.43   |
| JaNLI        |    81.10    |    80.92     |    73.41     |   77.98    |   77.44   |
| NU-MNLI      |    75.65    |    75.39     |    81.13     |   83.53    |   80.02   |

今回の実験では、JSNLIが最も高い性能を示しました。
また、興味深いこととして、MNLIを機械翻訳したのNU-MNLIでの性能が、比較的低めになっていることがわかります。
英語文埋め込みモデルではMNLIを加えることで性能が向上することが多いため、これは意外な結果となりました。

また、モデルごとにデータセットの順位をつけ、その平均を計算したものが以下の表です。

| Dataset      | Supervised |
| ------------ | ---------: |
| JSNLI        |      1.583 |
| NU-SNLI+MNLI |      2.000 |
| NU-SNLI      |      2.417 |
| NU-MNLI      |      4.375 |
| JaNLI        |      4.625 |

表からも、とりあえずJSNLIを選んでおけば間違いなさそうなことがわかります。
JaNLIは比較的低めの結果になっていますが、これはデータセットの規模が小さく、一般的なNLIデータセットとは少し違う目的で作成されていることが理由だと思われます。

#### Unsupervised SimCSE

以下の表は、教師なし設定において、モデルを`cl-tohoku/bert-large-japanese-v2`に固定した際の、データセットごとの結果です。

| Dataset   | JSICK (val) | JSICK (test) | JSTS (train) | JSTS (val) |   Avg.    |
| --------- | :---------: | :----------: | :----------: | :--------: | :-------: |
| Wikipedia |    79.63    |    79.40     |    77.18     |   80.28    |   78.95   |
| BCCWJ     |    79.60    |    79.45     |    76.71     |   80.83    |   79.00   |
| Wiki40B   |    79.54    |    79.14     |    77.18     |   81.00    | **79.11** |
| CC100     |    76.26    |    76.27     |    71.39     |   75.91    |   74.52   |

モデルとして`cl-tohoku/bert-large-japanese-v2`を使い場合、Wiki40Bが最も良い結果になりました。

また、モデルごとにデータセットの順位をつけ、その平均を計算しました。

| Dataset   | Unsupervised |
| --------- | -----------: |
| Wikipedia |        1.875 |
| BCCWJ     |        2.083 |
| Wiki40B   |        2.208 |
| CC100     |        3.833 |

上の表から、大雑把にですが、SimCSEの訓練にはWebコーパスよりWikipedia系を選んでおいた方がなんとなく良さそう、という傾向が読み取れます。

### Hyperparameters Ranking

次に、ハイパーパラメータについての順位も算出しました。
具体的には、バッチサイズごとに学習率の順位をつけ、その平均を計算しました。

| Supervised |  1e-5 |  3e-5 |  5e-5 |      Avg. |
| ---------: | ----: | ----: | ----: | --------: |
|         64 | 2.750 | 2.700 | 2.742 |     2.731 |
|        128 | 2.683 | 2.500 | 2.550 |     2.578 |
|        256 | 2.242 | 2.400 | 2.442 |     2.361 |
|        512 | 2.325 | 2.400 | 2.267 | **2.331** |


| Unsupervised |  1e-5 |  3e-5 |  5e-5 |      Avg. |
| -----------: | ----: | ----: | ----: | --------: |
|           64 | 1.344 | 2.083 | 2.500 |     1.976 |
|          128 | 1.969 | 1.656 | 1.854 | **1.826** |
|          256 | 2.990 | 2.792 | 2.375 |     2.719 |
|          512 | 3.698 | 3.469 | 3.271 |     3.479 |


表から、教師あり設定ではどのバッチサイズとして512を、教師なし設定ではバッチサイズとして64か128くらいを選んでおくのが良さそうということが読み取れます。


## 事前学習済みモデルの公開

最後に、公開した事前学習済み文埋め込みモデルについての詳細について説明します。

公開したモデルは、使いやすさ等の視点から選定したモデルについて、以下の表に示すハイパーパラメータでそれぞれのモデルをfine-tuningしたものです。
fine-tuning元のモデルとしては、教師あり設定では`cl-tohoku/bert-large-japanese-v2`、教師なし設定では`cl-tohoku/bert-base-japanese-v3`を用いています。
教師あり・教師なしの二つの設定についてモデルを公開しています。
3回実験を行い、最も開発セットでの性能が高いモデルを公開用のモデルとして選定しています。

| Model                                                                                     | Dataset |  LR   | Batch Size | STS Avg. |
| :---------------------------------------------------------------------------------------- | :------ | :---: | ---------: | :------: |
| [cl-nagoya/sup-simcse-ja-large](https://huggingface.co/cl-nagoya/sup-simcse-ja-large)     | JSNLI   | 5e-5  |        512 |  81.91   |
| [cl-nagoya/sup-simcse-ja-base](https://huggingface.co/cl-nagoya/sup-simcse-ja-base)       | JSNLI   | 5e-5  |        512 |  80.49   |
| [cl-nagoya/unsup-simcse-ja-large](https://huggingface.co/cl-nagoya/unsup-simcse-ja-large) | Wiki40b | 3e-5  |         64 |  79.60   |
| [cl-nagoya/unsup-simcse-ja-base](https://huggingface.co/cl-nagoya/unsup-simcse-ja-base)   | Wiki40b | 5e-5  |         64 |  77.48   |


また、公開したモデルと、既存の日本語対応文埋め込みモデルについて、評価結果を比較したものが以下の表になります。
評価には`src/evaluate.py`を用いています。

| Model                                                                                                                            | JSICK (val) | JSICK (test) | JSTS (train) | JSTS (val) |   Avg.    |
| -------------------------------------------------------------------------------------------------------------------------------- | :---------: | :----------: | :----------: | :--------: | :-------: |
| [cl-nagoya/sup-simcse-ja-large](https://huggingface.co/cl-nagoya/sup-simcse-ja-large)                                            |    84.36    |    83.05     |    79.61     |   83.07    | **81.91** |
| [cl-nagoya/sup-simcse-ja-base](https://huggingface.co/cl-nagoya/sup-simcse-ja-base)                                              |    83.62    |    82.75     |    77.86     |   80.86    |   80.49   |
| [cl-nagoya/unsup-simcse-ja-large](https://huggingface.co/cl-nagoya/unsup-simcse-ja-large)                                        |    79.89    |    79.62     |    77.77     |   81.40    |   79.60   |
| [cl-nagoya/unsup-simcse-ja-base](https://huggingface.co/cl-nagoya/unsup-simcse-ja-base)                                          |    79.15    |    79.01     |    74.48     |   78.95    |   77.48   |
|                                                                                                                                  |             |              |              |            |           |
| [pkshatech/GLuCoSE-base-ja](https://huggingface.co/pkshatech/GLuCoSE-base-ja)                                                    |    76.36    |    75.70     |    78.58     |   81.76    |   78.68   |
| [pkshatech/simcse-ja-bert-base-clcmlp](https://huggingface.co/pkshatech/simcse-ja-bert-base-clcmlp)                              |    74.47    |    73.46     |    78.05     |   80.14    |   77.21   |
| [colorfulscoop/sbert-base-ja](https://huggingface.co/colorfulscoop/sbert-base-ja)                                                |    67.19    |    65.73     |    74.16     |   74.24    |   71.38   |
| [sonoisa/sentence-luke-japanese-base-lite](https://huggingface.co/sonoisa/sentence-luke-japanese-base-lite)                      |    78.76    |    77.26     |    80.55     |   82.54    |   80.11   |
|                                                                                                                                  |             |              |              |            |           |
| [MU-Kindai/Japanese-SimCSE-BERT-large-sup](https://huggingface.co/MU-Kindai/Japanese-SimCSE-BERT-large-sup)                      |    77.06    |    77.48     |    70.83     |   75.83    |   74.71   |
| [MU-Kindai/Japanese-SimCSE-BERT-base-sup](https://huggingface.co/MU-Kindai/Japanese-SimCSE-BERT-base-sup)                        |    74.10    |    74.19     |    70.08     |   73.26    |   72.51   |
| [MU-Kindai/Japanese-SimCSE-BERT-large-unsup](https://huggingface.co/MU-Kindai/Japanese-SimCSE-BERT-large-unsup)                  |    77.63    |    77.69     |    74.05     |   77.77    |   76.50   |
| [MU-Kindai/Japanese-SimCSE-BERT-base-unsup](https://huggingface.co/MU-Kindai/Japanese-SimCSE-BERT-base-unsup)                    |    77.25    |    77.44     |    72.84     |   77.12    |   75.80   |
| [MU-Kindai/Japanese-MixCSE-BERT-base](https://huggingface.co/MU-Kindai/Japanese-MixCSE-BERT-base)                                |    76.72    |    76.94     |    72.40     |   76.23    |   75.19   |
| [MU-Kindai/Japanese-DiffCSE-BERT-base](https://huggingface.co/MU-Kindai/Japanese-DiffCSE-BERT-base)                              |    75.61    |    75.83     |    71.62     |   75.81    |   74.42   |
|                                                                                                                                  |             |              |              |            |           |
| [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE)                                                |    76.54    |    76.77     |    72.15     |   76.12    |   75.02   |
| [sentence-transformers/stsb-xlm-r-multilingual](https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual)            |    73.09    |    72.00     |    77.83     |   78.43    |   76.09   |
|                                                                                                                                  |             |              |              |            |           |
| [cl-tohoku/bert-large-japanese-v2 (Mean)](https://huggingface.co/cl-tohoku/bert-large-japanese-v2)                               |    67.06    |    67.15     |    66.72     |   70.68    |   68.18   |
| [studio-ousia/luke-japanese-large-lite (Mean)](https://huggingface.co/studio-ousia/luke-japanese-large-lite)                     |    62.23    |    60.90     |    65.41     |   68.02    |   64.78   |
| [studio-ousia/mluke-large-lite (Mean)](https://huggingface.co/studio-ousia/mluke-large-lite)                                     |    60.15    |    59.12     |    51.91     |   52.55    |   54.53   |
| [cl-tohoku/bert-base-japanese-v3 (Mean)](https://huggingface.co/cl-tohoku/bert-base-japanese-v3)                                 |    70.91    |    70.29     |    69.37     |   74.09    |   71.25   |
| [cl-tohoku/bert-base-japanese-v2 (Mean)](https://huggingface.co/cl-tohoku/bert-base-japanese-v2)                                 |    70.49    |    70.06     |    66.12     |   70.66    |   68.95   |
| [cl-tohoku/bert-base-japanese-whole-word-masking (Mean)](https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking) |    69.57    |    69.17     |    63.20     |   67.37    |   66.58   |
|                                                                                                                                  |             |              |              |            |           |
| [cl-tohoku/bert-large-japanese-v2 (CLS)](https://huggingface.co/cl-tohoku/bert-large-japanese-v2)                                |    46.66    |    47.02     |    54.13     |   57.38    |   52.84   |
| [cl-tohoku/bert-base-japanese-v3 (CLS)](https://huggingface.co/cl-tohoku/bert-base-japanese-v3)                                  |    51.37    |    51.91     |    58.49     |   62.96    |   57.79   |
|                                                                                                                                  |             |              |              |            |           |
| [text-embedding-ada-002](https://platform.openai.com/docs/api-reference/embeddings)                                              |    79.31    |    78.95     |    74.52     |   79.01    |   77.49   |

表から、全体として今回公開したモデルが最もよい性能を示していることがわかります。
また、OpenAIのtext-embedding-ada-002よりもより性能になっている点は注目に値します。

注意として、PKSHA社の文埋め込みモデルはJSTSの開発セットを訓練中の開発セットとして利用しているので、本実験の結果とは直接比較できません。
また、この評価結果はSTSタスクに限定されたものであり、情報検索タスクなど異なるタスクでの汎用性を保証するものではありません。
その点についてもご注意ください。

## まとめ

- SimCSEをベースとした日本語文埋め込みモデルについて網羅的に評価・検証を行いました。
- 実験の結果得られた良さそうなモデル4種類をHuggingFace上に公開しました。
- 日本語文埋め込みモデルのベースとしては以下を選んでおくと良さそうです。
  - Base
    - [cl-tohoku/bert-base-japanese-v3](https://huggingface.co/cl-tohoku/bert-base-japanese-v3)
    - [studio-ousia/luke-japanese-base-lite](https://huggingface.co/studio-ousia/luke-japanese-base-lite)
  - Large
    - [nlp-waseda/roberta-large-japanese](https://huggingface.co/nlp-waseda/roberta-large-japanese)
    - [cl-tohoku/bert-large-japanese-v2](https://huggingface.co/cl-tohoku/bert-large-japanese-v2)
    - [studio-ousia/luke-japanese-large-lite](https://huggingface.co/studio-ousia/luke-japanese-large-lite)
- 教師あり設定ではJSNLIを使うのが良さそうです。
  - NU-MNLIを加えても性能が向上しなかった点は興味深く、以下の要因がありそうです。
    1. 翻訳品質が悪い
    2. 高品質な翻訳は必ずしもSTSの性能に寄与しない
    3. ベンチマークデータセットの規模・多様性が不足している
    4. 運が悪かった
- 教師なし設定ではWikipediaを適当に使っておくのが良さそうです。
  - CC100の性能が悪かったので、綺麗さも大事そうです。
  - データセットごとに大きな違いはなさそうなので、綺麗であればなんでも良いかもしれません。

## 参考文献

- https://tech.yellowback.net/posts/sentence-transformers-japanese-models
- https://github.com/oshizo/JapaneseEmbeddingEval


```bibtex
@misc{
  hayato-tsukagoshi-2023-simple-simcse-ja,
  author = {Hayato Tsukagoshi},
  title = {Japanese Simple-SimCSE},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hppRC/simple-simcse-ja}}
}
```