import matplotlib.pyplot as plt

# Nhãn iteration muốn hiển thị
x_labels = [10, 12, 14, 16, 18, 20, 30]
x_pos = list(range(len(x_labels)))   # [0, 1, 2, 3, 4, 5, 6]

# Dữ liệu SSIM
nystrom_x = [1, 2, 3, 4, 5]   # tương ứng 12, 14, 16, 18, 20
nystrom_ssim = [0.9692, 0.9711, 0.9718, 0.9723, 0.9718]

learn_x = [1, 2, 3, 4, 5, 6]  # tương ứng 12, 14, 16, 18, 20, 30
learn_ssim = [0.9675, 0.9688, 0.9702, 0.9582, 0.9708, 0.9410]

# RegFormer tại iteration = 10
regformer_x = 0
regformer_ssim = 0.9680

plt.figure(figsize=(7, 5))

# Vẽ đường LEARN baseline
plt.plot(learn_x, learn_ssim,
         label='LEARN', marker='^', linestyle='-', color='royalblue', linewidth=2)

# Vẽ đường Nyströmformer
plt.plot(nystrom_x, nystrom_ssim,
         label='LEARN + Nyströmformer', marker='o', linestyle='-', color='red', linewidth=2)

# Vẽ đường ngang RegFormer
plt.plot([x_pos[0], x_pos[-1]], [regformer_ssim, regformer_ssim],
         linestyle='--', color='green', label='RegFormer', linewidth=1.5)

# Vẽ điểm RegFormer tại iteration=10
plt.scatter([regformer_x], [regformer_ssim], color='green', marker='s', zorder=5)

plt.xlabel('Iterations', fontsize=12)
plt.ylabel('SSIM', fontsize=12)

# Gắn nhãn trục x theo iteration thật
plt.xticks(x_pos, x_labels)

# Giới hạn trục x cho gọn
plt.xlim(-0.3, len(x_labels) - 0.7)

plt.legend(fontsize=10, loc='lower left')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('Ablation_Study_ssim.png', dpi=150)
plt.show()