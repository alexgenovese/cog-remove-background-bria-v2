import os, sys, tqdm
from huggingface_hub import login
from transformers import AutoModelForImageSegmentation
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
MODEL_CACHE = "model-cache/"
logged_in = False 

def login_hf():
    if logged_in is False:
        login( token = HF_TOKEN )
    
    return True

def start_download():
    if not os.path.exists(MODEL_CACHE):
        os.makedirs(MODEL_CACHE)
        # not exists directory, so not exists weights
        model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
        model.save_pretrained(MODEL_CACHE)


def download_weights(): 
    print("-----> Start caching models...")
    with tqdm(total=100, desc="Creating cache") as pbar:
        login_hf()
        pbar.update(25)

        start_download()
        pbar.update(75)

    print("-----> Caching completed!")


if __name__ == "__main__":
    download_weights() 