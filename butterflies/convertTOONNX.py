import torch
import torch.onnx
from model import CNNModel

# Khởi tạo mô hình
model = CNNModel()

# Tải checkpoint
checkpoint = torch.load('model.pth', map_location=torch.device('cpu'))

# Tải trọng số mô hình từ checkpoint
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Chuyển mô hình sang chế độ đánh giá


dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(model, dummy_input, 'model.onnx', verbose=True, input_names=['input'], output_names=['output'])
