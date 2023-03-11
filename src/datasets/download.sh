mkdir -p ./data

function jsick() {
    git clone git@github.com:verypluming/JSICK.git
    mv ./JSICK ./data/jsick
}

function jglue() {
    git clone git@github.com:yahoojapan/JGLUE.git
    mv ./JGLUE/datasets/jnli-v1.1 ./data/jnli
    mv ./JGLUE/datasets/jsts-v1.1 ./data/jsts
    rm -rf ./JGLUE
}

function janli() {
    git clone git@github.com:verypluming/JaNLI.git
    mv ./JaNLI ./data/janli
}

function jsnli() {
    wget "https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JSNLI/jsnli_1.1.zip"
    unzip ./jsnli_1.1.zip
    rm ./jsnli_1.1.zip
    mv ./jsnli_1.1 ./data/jsnli
}

function snli() {
    wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    unzip snli_1.0.zip -d snli
    mv snli data/snli
    rm snli_1.0.zip
}

function mnli() {
    wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
    unzip multinli_1.0.zip -d multinli
    mv multinli data/mnli
    rm multinli_1.0.zip
}

# http://www.cl.ecei.tohoku.ac.jp/rite2/wiki_%E3%83%AA%E3%82%BD%E3%83%BC%E3%82%B9%E3%83%97%E3%83%BC%E3%83%AB.html
# https://www.anlp.jp/proceedings/annual_meeting/2008/pdf_dir/E5-4.pdf
function ku_rte() {
    wget -O "./ku-rte.txt" "http://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/rte/entail_evaluation_set.txt&name=entail_evaluation_set.txt"
    mkdir data/ku-rte
    mv ./ku-rte.txt ./data/ku-rte/raw.txt
}

for func in jsick jglue janli jsnli snli mnli ku_rte; do
    $func >/dev/null 2>&1 &
done

wait
