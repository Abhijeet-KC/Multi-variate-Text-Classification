# NLP Multi-Label Classification with BERT / DistilBERT

## Overview
This project implements **multi-label text classification** using **BERT** and **DistilBERT** with TensorFlow (TF-Keras) or PyTorch. It supports:

- Tokenization with Hugging Face `transformers`
- GPU acceleration (PyTorch / TensorFlow)
- Multi-label classification
- Training / validation evaluation and metrics

---

## Recommended Setup

### Option 1: Run Easily on Google Colab (Recommended)
Colab provides:

- Pre-configured **GPU / TPU**
- Compatible **TensorFlow, PyTorch, CUDA, and Transformers**
- Avoids local installation headaches (drivers, CUDA version conflicts)

**Steps:**

1. Open [Colab](https://colab.research.google.com/)
2. Upload your notebook or clone the repo
3. Go to `Runtime → Change runtime type` → Select `GPU` or `TPU`
4. Install required packages:
```python
!pip install -q transformers tf-keras sentencepiece
```
5. Run your training cells, everything should work out-of-the-box.

### Option 2: Run Locally with GPU (Version-Oriented Setup)

Colab abstracts all CUDA and driver issues, but if you want to run locally on your **GTX 1650**, follow this setup to ensure proper GPU acceleration and compatibility.

#### 1️⃣ Prerequisites
- **GPU**: NVIDIA GTX 1650 (or higher)
- **CUDA**: 12.2 (recommended for TensorFlow 2.16) or 12.1 (for PyTorch 2.3)
- **NVIDIA Studio / Game Ready Driver** installed and working
- **Python** >= 3.10

#### 2️⃣ Python Environment Setup
1. Create a virtual environment:
```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

#### 3️⃣ Install required packages
```bash
python -r requirements.txt
```
#### 4️⃣ Verify GPU detection
```python
# For PyTorch

import torch
if torch.cuda.is_available():
    print("CUDA is available. Using GPU for training.")
    print("Device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU for training.")
```
```python
# For Tensorflow

import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("TensorFlow sees GPU(s):", gpus)
else:
    print("TensorFlow does not detect GPU. Using CPU for training.")
```
<ol>
  <li>Ensures local GPU is properly leveraged.</li>
  <li>Version-aligned PyTorch/TensorFlow avoids CUDA mismatches.</li>
  <li>Smaller batch sizes or DistilBERT help manage VRAM limitations on GTX 1650.</li>
</ol>
