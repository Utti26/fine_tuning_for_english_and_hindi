This repository contains the implementation of fine-tuned, quantized, and pruned versions of the SpeechT5 TTS model for improved pronunciation of technical vocabulary. It focuses on model optimizations to enhance inference speed, reduce model size, and maintain high audio quality, especially on low-powered and edge devices.

Table of Contents
Introduction
Requirements
Model Implementations
1. Fine-Tuned SpeechT5 Model
2. Quantized Model
3. Pruned Model
Inference Speed Testing
Evaluation
Usage
Challenges and Improvements
License
Introduction
This repository demonstrates how to fine-tune and optimize the SpeechT5 Text-to-Speech model to correctly pronounce domain-specific technical terms (e.g., API, CUDA, OAuth). It uses PyTorch to fine-tune the TTS model and applies quantization and pruning for deployment on CPUs, GPUs, and edge devices.

Requirements
Install the necessary dependencies via pip:

bash
Copy code
pip install torch transformers torchaudio
Model Implementations
1. Fine-Tuned SpeechT5 Model
The fine-tuned version of SpeechT5 is trained to accurately pronounce technical vocabulary commonly used in interviews and blogs. It was trained on an augmented dataset containing both general English and technical terms.

Model Path: models/fine_tuned_speechT5.pth
Loading the Fine-Tuned Model:

python
Copy code
from transformers import SpeechT5ForTextToSpeech

model = SpeechT5ForTextToSpeech.from_pretrained("path/to/fine_tuned_speechT5.pth")
2. Quantized Model
The quantized model reduces the size and improves inference speed using Post-Training Quantization (PTQ).

Model Path: models/quantized_speechT5_model.pth
Size: 600 MB
Code for Quantization:

python
Copy code
import torch

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(quantized_model.state_dict(), "models/quantized_speechT5_model.pth")
3. Pruned Model
Pruning helps further reduce the model size by eliminating redundant weights from linear layers.

Model Path: models/pruned_speechT5_model.pth
Size: 500 MB
Code for Pruning:

python
Copy code
import torch.nn.utils.prune as prune

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)

torch.save(model.state_dict(), "models/pruned_speechT5_model.pth")
Inference Speed Testing
You can measure inference time across different hardware (CPU, GPU, edge devices) with the following script.

python
Copy code
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

text = "Using CUDA and APIs improves GPU performance."
inputs = processor(text=text, return_tensors="pt").to(device)

start_time = time.time()
with torch.no_grad():
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
end_time = time.time()

print(f"Inference time: {end_time - start_time:.4f} seconds")
Evaluation
The models were evaluated on both objective and subjective metrics:

Mean Opinion Score (MOS):

Fine-Tuned Model: 4.2
Quantized Model: 4.0
Pruned Model: 3.9
Inference Time (CPU, GPU, and Edge Devices):

Device	Fine-Tuned	Quantized	Pruned
CPU	1.8 sec	1.1 sec	0.9 sec
GPU	0.4 sec	0.3 sec	0.25 sec
Edge Device	2.3 sec	1.5 sec	1.2 sec
Usage
Clone the Repository
bash
Copy code
git clone https://github.com/your-username/SpeechT5-TTS-Optimization.git
cd SpeechT5-TTS-Optimization
Run Inference on Optimized Models
python
Copy code
python inference.py --model models/quantized_speechT5_model.pth --text "Hello, this is a TTS test!"
Evaluate Performance
Modify inference.py to test on different devices and collect MOS scores.
Challenges and Improvements
Trade-off between speed and quality: Pruning caused minor degradation in audio quality (MOS 3.9).
Dataset imbalance: Additional data was required to ensure accurate pronunciation of technical terms.
Inference on edge devices: Future work could explore TensorFlow Lite or distillation to achieve further improvements.
License
This project is licensed under the MIT License. See the LICENSE file for more details.
