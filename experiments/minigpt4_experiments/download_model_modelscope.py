from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('shakechen/Llama-2-7b-chat-hf', cache_dir='models')