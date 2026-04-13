import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu
iterations_extended = [12, 14, 16, 18, 20]
nystrom_psnr_extended = [40.1634, 40.2132, 40.4193, 40.6327, 40.4704]

# LEARN baseline
learn_iterations = [12, 14, 16, 18, 20]
learn_psnr = [39.8808, 40.1522, 40.3055, 38.3685, 40.4512]

# Giá trị RegFormer chỉ có ở iterations=10
regformer_iter_single = 10
regformer_psnr_single = 40.2604

learn_iter_single = 30
learn_psnr_single = 36.2816

plt.figure(figsize=(7, 5))

# Vẽ đường LEARN baseline
plt.plot(learn_iterations, learn_psnr,
         label='LEARN', marker='^', linestyle='-', color='royalblue', linewidth=2)

# Vẽ đường Nystromformer
plt.plot(iterations_extended, nystrom_psnr_extended,
         label='LEARN + Nyströmformer', marker='o', linestyle='-', color='red', linewidth=2)

# Vẽ đường ngang RegFormer
plt.plot([9, 31], [regformer_psnr_single, regformer_psnr_single],
         linestyle='--', color='green', label='RegFormer', linewidth=1.5)

# Vẽ điểm RegFormer tại iterations=10
plt.scatter([regformer_iter_single], [regformer_psnr_single], color='green', marker='s', zorder=5)

# Vẽ đường ngang learn
plt.plot([9, 31], [learn_psnr_single, learn_psnr_single],
         linestyle='--', color='royalblue', label='LEARN (Baseline, iter=30)', linewidth=1.5)

# Vẽ điểm learn tại iterations=30
plt.scatter([learn_iter_single], [learn_psnr_single], color='royalblue', marker='^', zorder=5)

plt.xlabel('Iterations', fontsize=12)
plt.ylabel('PSNR (dB)', fontsize=12)
plt.xticks(range(10, 32, 2))
plt.legend(fontsize=10)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('Ablation_Study_psnr_v2.png', dpi=150)
plt.show()

