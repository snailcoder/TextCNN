## Train the model
Train a sentiment model based on Ctrip dataset for example.
```bash
python train.py ../CtripSentimentCorp.csv ctrip ./models
```

## Export to ONNX
Use torch.onnx.export to export torch model to ONNX. Set the 'dynamic_axes' argument to accept dynamic batch size and sequence length for deploying. See https://pytorch.org/docs/stable/onnx.html for details.
```python
import torch
import onnx

vocab = torch.load('./models/vocab.pth')
model = torch.load('./models/best_model.pth')
model.eval()

scripted = torch.jit.script(model)

dummy_input = torch.randint(low=0, high=len(vocab), size=(1, 30))

torch.onnx.export(scripted, dummy_input, 'textcnn.onnx', verbose=True,
                  input_names=['input'], output_names=['output'], opset_version=11,
                  dynamic_axes={
                    'input': {0: 'batch_size', 1: 'seq_len'},
                    'output': {0: 'batch_size'}})
```

