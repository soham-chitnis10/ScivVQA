import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ckpt_path = "IDEA-FinAI/chartmoe"
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ckpt_path, trust_remote_code=True).bfloat16().eval()
from dataset import SciVQADataset

# train_dataset = SciVQADataset("train_2025-03-27_18-34-44.json", "/media/soham/70DE3B11DE3ACEDA/Soham's_windows/Soham/NYU/Research/SciVQA/images_train")

