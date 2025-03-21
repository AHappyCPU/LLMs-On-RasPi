# Save as download_pythia_410m.py
import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)

def download_pythia_410m():
    """Download Pythia-410M model files"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--EleutherAI--pythia-410m")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download config and tokenizer
    base_url = "https://huggingface.co/EleutherAI/pythia-410m/resolve/main"
    files_to_download = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors"
    ]
    
    for file in files_to_download:
        download_url = f"{base_url}/{file}"
        download_path = os.path.join(cache_dir, file)
        
        if not os.path.exists(download_path):
            print(f"Downloading {file}...")
            download_file(download_url, download_path)
        else:
            print(f"{file} already exists, skipping.")
    
    print("Download complete! Now you can run the conversion script.")

if __name__ == "__main__":
    download_pythia_410m()