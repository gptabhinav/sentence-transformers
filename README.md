# Sentence Transformers

This was very specific to my GPU, so you might have to do setup properly.
But that should be okay, as long as you follow the steps below:

### System Requirements
- NVIDIA GPU with CUDA 6.1+.
- NVIDIA drivers >= 450.80.02 (check with nvidia-smi). Most GPU systems have compatible drivers.

### Setup with Conda
1. Install Miniconda: its to install conda, which is a package and environment management system that allows me to ensure that you have the right cuda environment, right python version and python dependencies. Following the steps here should be enough -- https://docs.conda.io/en/latest/miniconda.html
2. Create and activate the environment:
```bash
conda env create -f environment.yml
conda activate myproject
```
3. Verify PyTorch and Cuda setup is okay:
```python
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
```

### Testing everything out with an Example:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["This is a test sentence", "Another sentence"]
embeddings = model.encode(sentences)
print("Embeddings shape:", embeddings.shape)
```

if everything worked fine till here, noice!
but, if CUDA fails, use  `device='cpu'` in `SentenceTransformer` 
still facing issues, please report an issue
