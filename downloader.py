from huggingface_hub import snapshot_download

model_id = 'Qwen/Qwen2.5-VL-3B-Instruct'
local_dir = f'./{model_id.split("/")[-1]}'

snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)