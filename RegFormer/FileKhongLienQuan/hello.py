import torch
from torch.utils import cpp_extension

# Kiểm tra đường dẫn của cpp_extension
include_paths = cpp_extension.include_paths()
print(cpp_extension.include_paths())

# Kiểm tra xem CUDA có sẵn không
cuda_available = torch.cuda.is_available()
print(f"CUDA is available: {cuda_available}")
