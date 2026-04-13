import matplotlib.pyplot as plt

# Nhãn iteration muốn hiển thị
x_labels = [10, 12, 14, 16, 18, 20, 30]
x_pos = list(range(len(x_labels)))   # [0, 1, 2, 3, 4, 5, 6]

# Dữ liệu
learn_x = [1, 2, 3, 4, 5, 6]   # tương ứng 12, 14, 16, 18, 20, 30
learn_psnr = [39.8808, 40.1522, 40.3055, 38.3685, 40.4512, 36.2816]

nystrom_x = [1, 2, 3, 4, 5]    # tương ứng 12, 14, 16, 18, 20
nystrom_psnr = [40.1634, 40.2132, 40.4193, 40.6327, 40.4704]

regformer_x = 0                # tương ứng 10
regformer_psnr = 40.2604

plt.figure(figsize=(7, 5))

# LEARN
plt.plot(learn_x, learn_psnr,
         label='LEARN', marker='^', linestyle='-', color='royalblue', linewidth=2)

# LEARN + Nyströmformer
plt.plot(nystrom_x, nystrom_psnr,
         label='LEARN + Nyströmformer', marker='o', linestyle='-', color='red', linewidth=2)

# RegFormer
plt.plot([x_pos[0], x_pos[-1]], [regformer_psnr, regformer_psnr],
         linestyle='--', color='green', label='RegFormer', linewidth=1.5)
plt.scatter([regformer_x], [regformer_psnr], color='green', marker='s', zorder=5)

plt.xlabel('Iterations', fontsize=12)
plt.ylabel('PSNR (dB)', fontsize=12)

# Gắn nhãn trục x theo iteration thật
plt.xticks(x_pos, x_labels)

plt.xlim(-0.3, len(x_labels) - 0.7)

plt.legend(fontsize=10)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('Ablation_Study_psnr.png', dpi=150)
plt.show()