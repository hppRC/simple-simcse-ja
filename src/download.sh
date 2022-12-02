mkdir -p ./datasets/sts

function jsick() {
    git clone git@github.com:verypluming/JSICK.git
    mv ./JSICK ./datasets/sts/jsick
}

function jglue() {
    git clone git@github.com:yahoojapan/JGLUE.git
    mv ./JGLUE/datasets/jnli-v1.1 ./datasets/jnli
    mv ./JGLUE/datasets/jsts-v1.1 ./datasets/sts/jsts
    rm -rf ./JGLUE
}

function janli(){
    git clone git@github.com:verypluming/JaNLI.git
    mv ./JaNLI ./datasets/janli
}


function jsnli() {
    wget "https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JSNLI/jsnli_1.1.zip"
    unzip ./jsnli_1.1.zip
    rm ./jsnli_1.1.zip
    mv ./jsnli_1.1 ./datasets/jsnli
}


for func in jsick jglue janli jsnli; do
    $func > /dev/null 2>&1 &
done

wait