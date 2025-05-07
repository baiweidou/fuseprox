import torch
import torch.nn as nn
import torch.nn.functional as F
from .Resnet import resnet18 as resnet
import numpy as np
from torch.nn import init
from .Mixer_mlp import Res_MLP
from .modules.transformer import TransformerEncoder



class MultiModalNet(nn.Module):
    def __init__(self, num_classes=5, Branch='All', 
                 fusion_dim=128, num_heads=4, num_layers=2, attn_dropout=0.1):
        super().__init__()

        # 特征维度 (原始提取器输出维度)
        self.img_dim = 512
        self.gru_hidden_dim = 16
        self.static_input_dim = 35 
        
        # 各模态特征提取器
        self.img_net = resnet()
        self.gru = nn.GRU(input_size=4, hidden_size=self.gru_hidden_dim, batch_first=True)
        self.static_net = Res_MLP(input_dim=self.static_input_dim)
        # 获取静态网络的输出维度 - 根据错误信息，实际输出维度是 35
        # self.static_dim = getattr(self.static_net, 'output_dim', 10) # 旧代码，可能导致维度错误
        self.actual_static_dim = 35 # 假设 Res_MLP 输出维度为 35

        self.Branch = Branch
        self.fusion_dim = fusion_dim

        # --- 特征投影层 (将各模态投影到共同维度 fusion_dim) ---
        self.img_proj = nn.Linear(self.img_dim, self.fusion_dim)
        self.seq_proj = nn.Linear(self.gru_hidden_dim, self.fusion_dim) # GRU 输出 (last hidden state)
        # 使用正确的输入维度 (actual_static_dim) 和输出维度 (fusion_dim)
        self.static_proj = nn.Linear(self.actual_static_dim, self.fusion_dim)

        # --- Transformer 融合模块 ---
        # 定义模态间的交互: 例如 img->seq, img->static
        self.img_seq_transformer = TransformerEncoder(
            embed_dim=self.fusion_dim, 
            num_heads=num_heads, 
            layers=num_layers, 
            attn_dropout=attn_dropout
        )
        self.img_static_transformer = TransformerEncoder(
            embed_dim=self.fusion_dim,
            num_heads=num_heads,
            layers=num_layers,
            attn_dropout=attn_dropout
        )
        # 可以根据需要添加更多交互 TransformerEncoder, 如 seq_static_transformer

        # --- 计算最终分类器输入维度 ---
        # 融合策略: 拼接所有投影后的特征 + 所有Transformer交互后的特征
        # (proj_img + proj_seq + proj_static) + (img_seq_out + img_static_out)
        # 注意：这里假设了 Branch == 'All' 的情况
        # 如果需要支持其他 Branch，需要在这里或 forward 中添加逻辑来调整 feature_dims
        if self.Branch == 'All':
             # proj_img, proj_seq, proj_static 各 fusion_dim
             # img_seq_out, img_static_out 各 fusion_dim
            feature_dims = self.fusion_dim * 3 + self.fusion_dim * 2 
        elif self.Branch == 'CNN':
            # 仅使用原始图像特征 - 需要确保 FC 层能处理 img_dim
            # 或者也投影：feature_dims = self.fusion_dim
            feature_dims = self.img_dim # 保持原始维度，或修改
        elif self.Branch == 'Time':
            # 仅使用 GRU 特征
            feature_dims = self.gru_hidden_dim # 保持原始维度，或修改
        elif self.Branch == 'Static':
            # 仅使用静态特征
            feature_dims = self.actual_static_dim # 保持原始维度，或修改
        elif self.Branch == 'CNN+Time':
             # 示例：融合 CNN 和 Time (投影后 + Transformer交互)
             feature_dims = self.fusion_dim * 2 + self.fusion_dim # (proj_img + proj_seq) + img_seq_out
        elif self.Branch == 'Time+Static':
            # 示例：融合 Time 和 Static (需要添加 seq_static_transformer)
             # feature_dims = self.fusion_dim * 2 + self.fusion_dim 
             feature_dims = self.fusion_dim * 2 # 假设只拼接投影后的
        elif self.Branch == 'CNN+Static':
             # 示例：融合 CNN 和 Static (投影后 + Transformer交互)
            feature_dims = self.fusion_dim * 2 + self.fusion_dim # (proj_img + proj_static) + img_static_out
        else:
            raise ValueError(f"Unsupported Branch type: {self.Branch}")
        
        # --- 最终分类器 --- 
        self.fc = nn.Sequential(
            nn.Linear(feature_dims, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # 初始化权重 (可选)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, img, sequence, static):
        batch_size = img.size(0)

        # 1. 特征提取
        img_feat = self.img_net(img) # (B, img_dim)
        gru_out, _ = self.gru(sequence) # (B, seq_len, gru_hidden_dim)
        seq_feat = gru_out[:, -1, :] # 取最后时间步 (B, gru_hidden_dim)
        static_feat = self.static_net(static) # (B, static_dim)

        # --- 根据 Branch 选择特征 --- 
        if self.Branch == 'CNN':
            # 可能需要投影或直接使用 img_feat
            final_features = img_feat # 或者 self.img_proj(img_feat)
        elif self.Branch == 'Time':
            final_features = seq_feat # 或者 self.seq_proj(seq_feat)
        elif self.Branch == 'Static':
            final_features = static_feat # 或者 self.static_proj(static_feat)
        # --- 处理多模态分支 --- 
        else:
            # 2. 特征投影
            proj_img = F.relu(self.img_proj(img_feat))
            proj_seq = F.relu(self.seq_proj(seq_feat))
            proj_static = F.relu(self.static_proj(static_feat))

            # 3. 调整形状以适应 TransformerEncoder: (SeqLen, Batch, EmbedDim)
            # 将每个模态视为长度为 1 的序列
            tf_img = proj_img.unsqueeze(0)     # (1, B, fusion_dim)
            tf_seq = proj_seq.unsqueeze(0)     # (1, B, fusion_dim)
            tf_static = proj_static.unsqueeze(0) # (1, B, fusion_dim)

            # 4. Transformer 跨模态交互
            # (Query, Key, Value)
            img_seq_out = self.img_seq_transformer(tf_img, tf_seq, tf_seq) # (1, B, fusion_dim)
            img_static_out = self.img_static_transformer(tf_img, tf_static, tf_static) # (1, B, fusion_dim)
            # 如果需要其他交互 (如 seq <-> static), 在这里添加
            # seq_static_out = self.seq_static_transformer(tf_seq, tf_static, tf_static)
            
            # 移除多余的 SeqLen 维度
            img_seq_out = img_seq_out.squeeze(0) # (B, fusion_dim)
            img_static_out = img_static_out.squeeze(0) # (B, fusion_dim)
            # seq_static_out = seq_static_out.squeeze(0)

            # 5. 组合特征 (根据 Branch 决定最终特征)
            if self.Branch == 'All':
                final_features = torch.cat([
                    proj_img, proj_seq, proj_static, 
                    img_seq_out, img_static_out
                ], dim=1)
            elif self.Branch == 'CNN+Time':
                final_features = torch.cat([proj_img, proj_seq, img_seq_out], dim=1)
            elif self.Branch == 'Time+Static':
                # final_features = torch.cat([proj_seq, proj_static, seq_static_out], dim=1)
                final_features = torch.cat([proj_seq, proj_static], dim=1) # 假设只拼接投影
            elif self.Branch == 'CNN+Static':
                final_features = torch.cat([proj_img, proj_static, img_static_out], dim=1)
            else:
                 # Fallback or error for unhandled multi-modal branches
                 raise ValueError(f"Unhandled multi-modal Branch: {self.Branch}")

        # 6. 最终分类
        # 检查维度是否匹配 (可选，用于调试)
        expected_dim = self.fc[0].in_features
        if final_features.shape[1] != expected_dim:
             print(f"Error: Feature dimension mismatch before FC layer! Branch='{self.Branch}'. Expected {expected_dim}, got {final_features.shape[1]}. Check __init__ feature_dims calculation.")
             # 这里应该抛出错误或进行处理，而不是继续
             raise ValueError("Dimension mismatch") 
             
        output = self.fc(final_features)
        return output


# 使用示例
if __name__ == "__main__":
    # 超参数
    static_feature_size = 35  # 根据实际静态特征维度修改
    num_classes = 5  # 根据分类任务修改

    # 初始化模型
    model = MultiModalNet(use_dynamic_fusion=True, use_cross_attention=True)

    # 模拟输入数据
    batch_size = 4
    img_input = torch.randn(batch_size, 30, 32, 32)  # 图像输入
    seq_input = torch.randn(batch_size, 30, 4)  # 时序输入
    static_input = torch.randn(batch_size, static_feature_size)  # 静态特征

    # 前向传播
    output, (attn_weights_dict, fusion_weights) = model(img_input, seq_input, static_input)

    print("Output shape:", output.shape)
    print("Time attention weights shape:", attn_weights_dict['time_attn'].shape)
    if fusion_weights is not None:
        print("Fusion weights shape:", fusion_weights.shape)