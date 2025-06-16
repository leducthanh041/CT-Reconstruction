import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Dữ liệu
iterations_extended = [12, 14, 16, 18, 20, 22, 24]
nystrom_psnr_extended = [0.9692, 0.9711,	0.9718,	0.9723,	0.9718,	0.9676,	0.9722]

# Giá trị RegFormer chỉ có ở iterations=10
regformer_iteration = 10
regformer_psnr = 0.9680

plt.figure(figsize=(7, 4))

# Vẽ đường Nystromformer
plt.plot(iterations_extended, nystrom_psnr_extended, label='Nystromformer', marker='o', linestyle='-', color='red')

# Vẽ đường ngang RegFormer
plt.plot([9, 25], [regformer_psnr, regformer_psnr], 
         linestyle='--', color='green', label='RegFormer')

# Vẽ điểm RegFormer tại iterations=10
plt.scatter([regformer_iteration], [regformer_psnr], color='green', marker='s', zorder=5)

# Tuỳ chỉnh legend
custom_lines = [
    Line2D([0], [0], linestyle='-', color='red', marker='o', markersize=10, label='LEARN+Nyströmformer'),
    Line2D([0], [0], linestyle='--', color='green', marker='s', markersize=10, label='RegFormer')
]

plt.xlabel('Iterations')
plt.ylabel('SSIM')
plt.legend(handles=custom_lines)
plt.grid(True)
plt.savefig('Ablation_Study_ssim.png')
plt.show()
