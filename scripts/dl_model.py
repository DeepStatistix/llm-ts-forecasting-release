import os, sys
os.environ.setdefault("HF_HOME", r"E:\hf_cache")
from huggingface_hub import snapshot_download
pats = sys.argv[2:] or None   # e.g. "*.safetensors" "*.json" "tokenizer*" to skip duplicate weight formats
print(snapshot_download(sys.argv[1], allow_patterns=pats), flush=True)
print("DOWNLOAD_DONE", flush=True)
