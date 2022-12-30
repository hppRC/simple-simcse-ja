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

for func in jsick jglue janli jsnli; do
    $func >/dev/null 2>&1 &
done

wait
