from huggingface_hub import HfApi

# MODEL_PATH = "./outputs/sup-simcse/jsnli/cl-tohoku__bert-base-japanese-v3/2023-10-02/16-22-36"
# MODEL_PATH = "./outputs/sup-simcse/jsnli/cl-tohoku__bert-large-japanese-v2/2023-10-02/16-22-31"
# MODEL_PATH = "./outputs/unsup-simcse/wiki40b/cl-tohoku__bert-base-japanese-v3/2023-10-02/17-15-39"
MODEL_PATH = "./outputs/unsup-simcse/wiki40b/cl-tohoku__bert-large-japanese-v2/2023-10-02/17-26-28"

REPO_ID = "cl-nagoya/unsup-simcse-ja-large"

HfApi().create_repo(REPO_ID)
HfApi().upload_folder(folder_path=MODEL_PATH, repo_id=REPO_ID)
