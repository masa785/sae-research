import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())