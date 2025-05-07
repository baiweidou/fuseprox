# -*- coding:utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # 用于划分训练集和测试集
from sklearn.preprocessing import StandardScaler, OneHotEncoder # 用于数据标准化和独热编码
from tqdm import tqdm # 用于显示进度条
import warnings

warnings.filterwarnings("ignore", category=FutureWarning) # 忽略一些未来版本可能产生的警告

# --- 常量定义 ---
IMG_SIZE = 32 # 负载转换后图像的尺寸 (32x32)
N_PACKETS = 30 # 要处理的数据包/负载的数量（前30个）

# --- 辅助函数 (hex_to_image 保持不变) ---
def hex_to_image(hex_str, img_size=IMG_SIZE):
    """将代表字节的十六进制字符串转换为归一化的图像数组。"""
    try:
        target_length = img_size * img_size * 2
        hex_str = str(hex_str).strip('. ') # 清理可能的填充字符
        hex_str = hex_str.ljust(target_length, '0') # 右侧填充 '0' 使长度足够
        hex_str = hex_str[:target_length] # 截断过长的部分
        byte_array = np.array(
            # 每两个十六进制字符转换为一个 0-255 的整数
            [int(hex_str[i:i + 2], 16) for i in range(0, target_length, 2)],
            dtype=np.uint8 # 指定为8位无符号整数
        )
        # 重塑为图像尺寸，并将像素值归一化到 [0, 1]
        image = byte_array.reshape(img_size, img_size).astype(np.float32) / 255.0
        return image
    except Exception as e:
        # print(f"将十六进制转换为图像时出错: {e}. Hex: '{hex_str[:50]}...'") # 可选的调试输出
        # 返回全零数组作为占位符
        return np.zeros((img_size, img_size), dtype=np.float32)

# --- 主执行块 ---
if __name__ == '__main__':
    # --- 配置参数 ---
    data_inf_path = 'data/network_features_enhanced_random_sampled_2k.csv' # 输入CSV文件路径
    output_dir = 'data_processed' # 定义处理后数据的输出目录（建议用新目录）
    # 定义合并后的输出文件路径
    time_output_file = os.path.join(output_dir, 'time_features.npy')    # 时序特征输出文件
    static_output_file = os.path.join(output_dir, 'static_features.npy') # 静态特征输出文件
    imgs_output_file = os.path.join(output_dir, 'imgs_features.npy')    # 图像特征输出文件
    # 定义标签/索引文件路径
    train_file = os.path.join(output_dir, 'train.csv') # 训练集索引/标签文件
    val_file = os.path.join(output_dir, 'val.csv')     # 验证集索引/标签文件
    all_labels_file = os.path.join(output_dir, 'all_labels.csv') # 可选：保存所有标签和原始索引

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # --- 加载数据 ---
    print(f"从 {data_inf_path} 加载数据...")
    data_inf = pd.read_csv(data_inf_path)
    print(f"数据加载完成。形状: {data_inf.shape}")
    n_samples = len(data_inf) # 获取样本数量

    # 保留原始索引（如果需要），然后重置索引为 0 到 N-1，方便后续对齐
    data_inf['original_index'] = data_inf.index
    data_inf.reset_index(drop=True, inplace=True)

    # --- 特征定义 ---
    # 类别特征列名
    categorical_cols = ['protocol', 'ip_version', 'tls_version', 'cipher_suite', 'first_direction']
    # 数值特征列名
    numeric_cols = ['flow_duration', 'packet_rate', 'byte_rate', 'iat_mean',
                    'iat_std', 'iat_min', 'iat_max', 'iat_median',
                    'forward_ratio', 'forward_bytes_ratio']
    # 负载特征列名模式
    payload_columns = [f'payload_{i}' for i in range(1, N_PACKETS + 1)]
    # 时序特征列名（基础名称）
    time_series_base_cols = ['pkt_len', 'entropy', 'direction', 'pkt_time']
    # 生成所有时序特征的完整列名
    time_series_cols = [f'{base}_{i}' for i in range(1, N_PACKETS + 1) for base in time_series_base_cols]

    # --- 1. 处理静态特征 (向量化) ---
    print("处理静态特征...")
    # 对类别特征进行独热编码
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # 在整个数据集上拟合和转换
    categorical_data_encoded = encoder.fit_transform(data_inf[categorical_cols]) # 形状: (样本数, 编码后的特征数)
    print(f"  编码后的类别特征形状: {categorical_data_encoded.shape}")

    # 提取数值特征
    numeric_data = data_inf[numeric_cols].values.astype(np.float32) # 形状: (样本数, 数值特征数)
    print(f"  数值特征形状: {numeric_data.shape}")

    # 合并编码后的类别特征和数值特征
    # 注意：标准化将在训练/验证集划分之后进行
    static_features_combined = np.hstack([categorical_data_encoded, numeric_data]) # 形状: (样本数, 总静态特征数)
    print(f"  合并后的静态特征形状: {static_features_combined.shape}")

    # --- 2. 处理时序特征 ---
    print("处理时序特征...")
    # 检查并处理可能缺失的时序特征列
    missing_ts_cols = [col for col in time_series_cols if col not in data_inf.columns]
    if missing_ts_cols:
        print(f"  警告: 缺失以下时序特征列: {missing_ts_cols}。将用 0 填充。")
        for col in missing_ts_cols:
             data_inf[col] = 0 # 添加缺失列并填充0

    # 将所有时序数据提取到一个 NumPy 数组中
    ts_data_raw = data_inf[time_series_cols].values.astype(np.float32) # 形状: (样本数, N_PACKETS * 4)

    # 重塑数据: (样本数, N_PACKETS * 4) -> (样本数, N_PACKETS, 4)
    try:
        # 最后一个维度是每个时间步的特征数量 (len(time_series_base_cols))
        time_features = ts_data_raw.reshape(n_samples, N_PACKETS, len(time_series_base_cols))
        print(f"  时序特征形状: {time_features.shape}")
    except ValueError as e:
        print(f"  重塑时序数据时出错: {e}")
        print(f"  期望列数: {N_PACKETS * len(time_series_base_cols)}, 实际列数: {ts_data_raw.shape[1]}")
        # 根据需要处理错误，例如用零填充或抛出异常
        time_features = np.zeros((n_samples, N_PACKETS, len(time_series_base_cols)), dtype=np.float32)

    # --- 3. 处理负载特征 (图像) ---
    print("处理负载特征 (图像)...")
    # 检查并处理可能缺失的负载列
    missing_payload_cols = [col for col in payload_columns if col not in data_inf.columns]
    if missing_payload_cols:
         print(f"  警告: 缺失以下负载列: {missing_payload_cols}。将生成零图像。")
         for col in missing_payload_cols:
             data_inf[col] = '' # 填充空字符串，以便 hex_to_image 能处理

    # 对每个 payload 列应用 hex_to_image 函数
    payload_images_list = []
    for col in tqdm(payload_columns, desc="  转换十六进制负载"):
        # 使用 .fillna('') 处理可能的 NaN 值，防止 apply 出错
        # images_for_col 是一个 Pandas Series，其中每个元素是一个 (IMG_SIZE, IMG_SIZE) 的 NumPy 数组
        images_for_col = data_inf[col].fillna('').apply(hex_to_image)
        # 将这个 Series 中的所有 NumPy 数组堆叠起来，形成 (样本数, IMG_SIZE, IMG_SIZE) 的数组
        payload_images_list.append(np.stack(images_for_col.values))

    # 沿着新的维度（数据包/时间步维度，即 axis=1）堆叠列表中的数组
    # 结果形状: (样本数, N_PACKETS, IMG_SIZE, IMG_SIZE)
    imgs_features = np.stack(payload_images_list, axis=1)
    print(f"  负载图像特征形状: {imgs_features.shape}")

    # --- 4. 准备标签并划分数据 ---
    print("准备标签并划分数据...")
    labels = data_inf['label'].values.astype(np.int64) # 使用 int64 类型，适用于 PyTorch 的 CrossEntropyLoss
    # 使用重置后的索引 (0 到 N-1) 来进行划分
    indices = np.arange(n_samples)

    # 创建用于划分的 DataFrame，包含新的行索引和标签
    split_df = pd.DataFrame({'idx': indices, 'label': labels})

    # 划分训练集和验证集
    train_df, val_df = train_test_split(
        split_df,
        test_size=0.2,             # 验证集比例 20%
        random_state=42,           # 随机种子保证可复现
        shuffle=True,              # 划分前打乱数据
        # stratify 参数用于分层抽样，确保训练集和验证集中各类别比例与原始数据大致相同
        stratify=split_df['label'] if split_df['label'].nunique() > 1 else None
    )

    # 获取训练集和验证集的索引 (这些索引对应于大型 .npy 文件中的行号)
    train_indices = train_df['idx'].values
    val_indices = val_df['idx'].values

    # --- 5. 应用 StandardScaler (可选, 仅在训练数据上拟合) ---
    scaler = StandardScaler()
    # 确定合并后的静态特征中，数值特征部分的位置
    # (它们位于最后 numeric_data.shape[1] 列)
    num_numeric_features = numeric_data.shape[1]
    # 创建一个 slice 对象来选择数值特征列
    numeric_part_indices = slice(-num_numeric_features, None)

    print("在训练数据的数值特征上拟合 StandardScaler...")
    # 使用训练集的索引从合并后的静态特征中提取出训练数据的数值部分，并进行拟合
    scaler.fit(static_features_combined[train_indices, numeric_part_indices])

    print("应用 StandardScaler 到所有静态特征的数值部分...")
    # 将拟合好的 scaler 应用于整个数据集的数值特征部分
    static_features_combined[:, numeric_part_indices] = scaler.transform(
        static_features_combined[:, numeric_part_indices]
    )
    # 注意：通常不对独热编码产生的类别特征进行标准化

    # --- 6. 保存处理后的数据 (合并文件) ---
    print(f"保存处理后的数据到 {output_dir}...")

    # 保存特征数据 (所有样本保存在一个文件里)
    np.save(static_output_file, static_features_combined)
    print(f"  已保存静态特征到 {static_output_file}")
    np.save(time_output_file, time_features)
    print(f"  已保存时序特征到 {time_output_file}")
    np.save(imgs_output_file, imgs_features)
    print(f"  已保存图像特征到 {imgs_output_file}")

    # 保存训练集和验证集的索引/标签文件
    # 这些文件现在包含的是对应于大型 .npy 文件中行号的 'idx'
    train_df[['idx', 'label']].to_csv(train_file, index=False)
    print(f"  已保存训练集索引/标签到 {train_file}")
    val_df[['idx', 'label']].to_csv(val_file, index=False)
    print(f"  已保存验证集索引/标签到 {val_file}")

    # 可选：如果需要在其他地方使用原始索引，可以保存原始索引和标签的映射
    # data_inf[['original_index', 'label']].to_csv(all_labels_file, index=False)
    # print(f"  已保存所有原始索引/标签到 {all_labels_file}")

    print("数据处理完成。")

