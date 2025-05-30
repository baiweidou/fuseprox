import torch.nn as nn
import torch
from einops import rearrange
import math
import warnings

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class FA_Layer(nn.Module):
    # 确保 feature_dim 参数被正确使用
    def __init__(self, n_dim=1, feature_dim=10, dropout_prob=0.2): # feature_dim 现在会从 Res_MLP 传入
        super(FA_Layer, self).__init__()
        self.feature_linear = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=feature_dim * 3),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(feature_dim * 3, feature_dim) # 输出维度匹配 feature_dim
        )
        self.n_linear = nn.Sequential(
            nn.LayerNorm(n_dim),
            nn.Linear(in_features=n_dim, out_features=n_dim * 3),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(n_dim * 3, n_dim)
        )

    def forward(self, x):
        # x 的形状应为 (bs, n_dim, feature_dim)
        bs, n_dim, feature_dim = x.shape
        feature_inf = self.feature_linear(x) # 输入/输出维度匹配 feature_dim
        n_inf = x.permute(0, 2, 1)
        n_inf = self.n_linear(n_inf)
        n_inf = n_inf.permute(0, 2, 1)
        fuse_feature = x + n_inf + feature_inf # 输出形状 (bs, n_dim, feature_dim)
        return fuse_feature
class Res_MLP(nn.Module):
    # 修改 __init__ 以接受 input_dim
    def __init__(self, input_dim=10, num_layer=3, Out_flag=False, dropout_prob=0.1): # 添加 input_dim 参数
        super(Res_MLP, self).__init__()
        self.Out_flag = Out_flag
        self.input_dim = input_dim # 保存维度

        # 使用 self.input_dim 替换硬编码的 10
        self.encode = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim * 2), # 输入层使用 input_dim
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(self.input_dim * 2, self.input_dim), # 输出层也使用 input_dim
            nn.LayerNorm(self.input_dim)                   # LayerNorm 也使用 input_dim
        )

        # 在初始化 FA_Layer 时传递正确的 feature_dim
        self.FA = nn.ModuleList([FA_Layer(feature_dim=self.input_dim) for _ in range(num_layer)])

        # --- 注意：下面的 out_fc* 层是用于特定任务的，如果你的主模型不使用它们，可以忽略 ---
        # 如果你的分类任务直接使用 Res_MLP 的输出 (fuse_feature)，这些层可能不需要修改
        # 如果这些层也需要调整，请确保它们的输入维度基于 self.input_dim * 3 (如果它们处理 FA 输出)
        # 或者 self.input_dim (如果它们处理 encode 输出)
        self.out_fc = nn.Linear(self.input_dim, 2) # 示例：假设输出层需要调整
        self.out_fc_number = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim), # 示例
            nn.ReLU(),
            nn.Linear(self.input_dim, 1)
        )
        self.out_fc_rush = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim), # 示例
            nn.ReLU(),
            nn.Linear(self.input_dim, 1)
        )
        # --- 注意结束 ---

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # 确保输入 x 的维度是 (batch_size, self.input_dim)
        x = self.encode(x).unsqueeze(1)  # encode 输出现在是 (batch_size, 1, self.input_dim)
        for model in self.FA:
            fuse_feature = model(x)  # FA 层内部也应使用正确的维度
        fuse_feature = fuse_feature.squeeze(1)  # 输出维度是 (batch_size, self.input_dim)
        # --- 注意：根据你的 Res_MLP 用途，决定返回什么 ---
        # 如果你的主模型 MultiModalNet 使用 Res_MLP 的主要特征输出：
        return fuse_feature
        # 如果你的 Res_MLP 本身是一个独立模型，需要返回 out_fc* 的结果：
        # out_number = self.out_fc_number(fuse_feature)
        # out_rush = self.out_fc_rush(fuse_feature)
        # return out_number, out_rush
        # --- 注意结束 ---


if __name__ == '__main__':
    satellite = torch.rand(4,10)
    model = Res_MLP(Out_flag=True)
    out_number, out_rush = model(satellite)
    pass
