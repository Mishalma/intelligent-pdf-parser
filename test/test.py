# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("microsoft/layoutlmv3-large", torch_dtype="auto")