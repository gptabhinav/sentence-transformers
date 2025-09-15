from sentence_transformers import SentenceTransformer
import torch
import numpy

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
print("NumPy version:", numpy.__version__)

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["This is a test sentence", "Another sentence"]
embeddings = model.encode(sentences)
print("Embeddings shape:", embeddings.shape)
