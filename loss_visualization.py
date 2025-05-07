import matplotlib.pyplot as plt
import numpy as np

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei'] # 或者 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# 模拟参数
epochs = 100
epoch_range = np.arange(1, epochs + 1)
N_values = [5, 10, 20, 30, 40, 50]
colors = plt.cm.viridis(np.linspace(0, 1, len(N_values))) # 使用 colormap 生成不同颜色

plt.figure(figsize=(10, 6))

# 模拟并绘制每个 N 值的 LOSS 曲线
for i, N in enumerate(N_values):
    # 基础 LOSS 函数模型 (例如: 指数衰减 + 最终值 + 噪声)
    # 调整参数以匹配描述
    initial_loss = 2
    # 为所有N值增加基础的随机性，并根据N调整最终目标和噪声水平
    base_noise_multiplier = 0.03 # 基础噪声幅度因子，增加随机性
    if N == 5:
        final_loss_target = 0.52
        decay_rate = 0.08
        noise_level = base_noise_multiplier * 1.8 # N越小，相对波动可能更大
    elif N == 10:
        final_loss_target = 0.35
        decay_rate = 0.09
        noise_level = base_noise_multiplier * 1.5
    elif N == 20:
        final_loss_target = 0.18
        decay_rate = 0.10
        noise_level = base_noise_multiplier * 1.2
    elif N == 30:
        final_loss_target = 0.11
        decay_rate = 0.11
        noise_level = base_noise_multiplier * 1.0 # N=30 相对最稳定
    elif N == 40:
        final_loss_target = 0.12
        decay_rate = 0.10 # 收敛稍慢
        noise_level = base_noise_multiplier * 1.4 # 噪声开始增加
    else: # N = 50
        final_loss_target = 0.14
        decay_rate = 0.09 # 收敛更慢
        noise_level = base_noise_multiplier * 1.7 # 噪声更明显

    # 生成基础 LOSS 曲线
    base_loss = initial_loss * np.exp(-decay_rate * epoch_range) + final_loss_target

    # 生成贯穿始终的噪声，并根据 noise_level 调整幅度
    # 让噪声在后期稍微增大一点模拟训练后期可能的波动增加
    noise = np.random.normal(0, noise_level * (1 + 0.5 * epoch_range / epochs), epochs)

    # 将噪声加到基础 LOSS 曲线上
    loss = base_loss + noise

    # 避免LOSS因为噪声变得过低或为负
    loss = np.maximum(loss, final_loss_target * 0.5) # 限制最低值，防止噪声导致不合理下降
    loss = np.maximum(loss, 0.01) # 确保loss大于一个很小的值

    # 绘制曲线
    plt.plot(epoch_range, loss, color=colors[i], label=f'N = {N}', alpha=0.8) # 略微透明处理重叠部分

# 添加标题和标签
# plt.title('不同 N 值下的模拟训练 LOSS 曲线 (增加随机性)')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Number of Data Packets (N)")
plt.ylim(bottom=0) # LOSS 不应为负

# 显示图表
plt.show()

# 如果需要保存图表到文件
# plt.savefig('training_loss_curves_noisy.png') 