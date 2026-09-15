import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph


def create_length_mask(x, lengths):
    """
    Concise function for creating a length mask
    Args:
        x: input tensor (N, C, T, V) or (N, C, T)
        lengths: length tensor, can be a scalar, 1D tensor, or 2D tensor
    Returns:
        mask: boolean mask (N, T)
    """
    if x.dim() == 4:
        N, C, T, V = x.shape
    elif x.dim() == 3:
        N, C, T = x.shape
    else:
        raise ValueError(f"Unsupported input dimension: {x.dim()}")

    device = x.device

    if lengths is None:
        return torch.ones(N, T, device=device, dtype=torch.bool)

    if isinstance(lengths, (int, float)):

        lengths_tensor = torch.tensor([lengths] * N, device=device)
    else:

        lengths_flat = lengths.flatten()
        if len(lengths_flat) == 1:

            lengths_tensor = torch.full((N,), lengths_flat.item(), device=device)
        else:

            if len(lengths_flat) >= N:
                lengths_tensor = lengths_flat[:N].to(device)
            else:

                last_val = lengths_flat[-1].item()
                lengths_tensor = torch.tensor(
                    list(lengths_flat.cpu().numpy()) + [last_val] * (N - len(lengths_flat)),
                    device=device
                )

    time_range = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
    lengths_expanded = lengths_tensor.unsqueeze(1)  # (N, 1)
    mask = time_range < lengths_expanded  # (N, T)

    return mask



def create_and_expand_mask(x, lengths):
    mask = create_length_mask(x, lengths)
    mask_expanded = mask.unsqueeze(1).unsqueeze(-1).expand_as(x)
    return mask_expanded


def compute_adjusted_lengths(lengths, stride, T_orig, T_new, device):
    if lengths is None:
        return None

    adjusted_lengths = []

    for i in range(len(lengths) if lengths.dim() > 0 else 1):
        if lengths.dim() == 0:
            length_val = lengths.item()
        else:
            length_val = lengths[i]

        if stride > 1:
            adjusted_length = (length_val + stride - 1) // stride
        else:
            adjusted_length = length_val

        adjusted_length = min(adjusted_length, T_new)
        adjusted_length = max(1, adjusted_length)
        adjusted_lengths.append(adjusted_length)

    return torch.tensor(adjusted_lengths, device=device)


class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes, graph_args,
                 temporal_kernel_size=9, edge_importance_weighting=True, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        spatial_kernel_size = A.size(0)
        kernel_size = [(temporal_kernel_size, spatial_kernel_size),
                       (temporal_kernel_size, spatial_kernel_size),
                       (temporal_kernel_size, spatial_kernel_size)]

        self.feature_embedding = SkeletonFeatureEmbedding(residual_scale=0.1)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        # ST-GCN
        self.st_gcn_networks = nn.ModuleList((
            MaskAwareSTGCN(14, 64, kernel_size[0], 1, dropout=0.4, has_ca=False, **kwargs),
            MaskAwareSTGCN(64, 64, kernel_size[0], 1, dropout=0.4, has_ca=False, **kwargs),
            MaskAwareSTGCN(64, 64, kernel_size[0], 1, dropout=0.4, has_ca=False, **kwargs),

            MaskAwareSTGCN(64, 128, kernel_size[0], 2, dropout=0.4, has_ca=False, **kwargs),
            MaskAwareSTGCN(128, 128, kernel_size[1], 1, dropout=0.4, has_ca=False, **kwargs),
            MaskAwareSTGCN(128, 128, kernel_size[1], 1, dropout=0.4, has_ca=False, **kwargs),

            MaskAwareSTGCN(128, 256, kernel_size[1], 2, dropout=0.4, has_ca=False, **kwargs),
            MaskAwareSTGCN(256, 256, kernel_size[2], 1, dropout=0.4, has_ca=False, **kwargs),
            MaskAwareSTGCN(256, 256, kernel_size[2], 1, dropout=0.4, has_ca=False, **kwargs),
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.mask_pooling = MaskAwareGlobalPooling()

        # ---- Three layer feature concatenation ----
        selected_channels = [64, 128, 256]
        self.sum_channels = sum(selected_channels)          # 448


        self.fusion_bottleneck = nn.Conv2d(self.sum_channels, 256, kernel_size=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(256)
        self.fusion_relu = nn.ReLU(inplace=True)

        reduction = 8
        self.se_fusion = nn.Sequential(
            nn.Conv2d(256, 256 // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(256 // reduction, 256, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)



    def forward(self, x, lengths):
        N, C, T, V, M = x.size()

        # Feature embedding and data normalization
        x = x.permute(0, 4, 2, 3, 1).contiguous()
        x = x.view(N * M, T, V, C)
        embedded = self.feature_embedding(x)
        x = embedded.permute(0, 3, 2, 1).contiguous()
        x = x.view(N * M, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, M, C, V, T)
        x = x.permute(0, 1, 2, 4, 3).contiguous()
        x = x.view(N * M, C, T, V)

        multi_pooled = []

        for idx, (gcn, importance) in enumerate(zip(self.st_gcn_networks, self.edge_importance)):
            x, _, lengths = gcn(x, self.A * importance, lengths)

            if idx in [2, 5, 8]:
                pooled = self.mask_pooling(x, lengths)                 # (N*M, C, 1, V)
                pooled = pooled.view(N, M, -1, 1, V).mean(dim=1)       # (N, C, 1, V)
                multi_pooled.append(pooled)

        pooled_1 = F.normalize(multi_pooled[0], p=2, dim=1)  # (N, 64, 1, V)
        pooled_2 = F.normalize(multi_pooled[1], p=2, dim=1)  # (N, 128, 1, V)
        pooled_3 = F.normalize(multi_pooled[2], p=2, dim=1)  # (N, 256, 1, V)

        concat_feat = torch.cat([pooled_1, pooled_2, pooled_3], dim=1)  # (N, 448, 1, V)

        fused = self.fusion_bottleneck(concat_feat)               # (N, 256, 1, V)
        fused = self.fusion_bn(fused)                                  # (N, 256, 1, V)
        fused = self.fusion_relu(fused)                                # (N, 256, 1, V)


        fused = fused.mean(dim=-1, keepdim=True)  # (N, 256, 1, 1)
        f = fused.squeeze()                                            # (N, 256)


        out = self.fcn(fused)                                          # (N, num_classes, 1, 1)
        out = out.view(out.size(0), -1)                                # (N, num_classes)

        return out, f



class SkeletonFeatureEmbedding(nn.Module):
    """Embedding module for all features [gait(3) + move(8) + pos(1) + angle(2)] = 14 dimensions"""

    def __init__(self,
                 residual_scale=0.2,
                 gait_expansion=4,
                 move_expansion=4,
                 pos_expansion=4,
                 angle_expansion=4,
                 use_layer_norm=True):

        super().__init__()

        self.input_dim = 14  # 3+8+1+2
        self.residual_scale = residual_scale
        self.use_layer_norm = use_layer_norm

        gait_hidden = 3 * gait_expansion
        move_hidden = 8 * move_expansion
        pos_hidden = 1 * pos_expansion
        angle_hidden = 2 * angle_expansion

        self.gait_embed = self._build_embedding_block(
            in_dim=3,
            hidden_dim=gait_hidden,
            out_dim=3,
            name="gait"
        )

        self.move_embed = self._build_embedding_block(
            in_dim=8,
            hidden_dim=move_hidden,
            out_dim=8,
            name="move"
        )

        self.pos_embed = self._build_embedding_block(
            in_dim=1,
            hidden_dim=pos_hidden,
            out_dim=1,
            name="pos"
        )

        self.angle_embed = self._build_embedding_block(
            in_dim=2,
            hidden_dim=angle_hidden,
            out_dim=2,
            name="angle"
        )

        self.expansion_config = {
            'gait': gait_expansion,
            'move': move_expansion,
            'pos': pos_expansion,
            'angle': angle_expansion
        }


    def _build_embedding_block(self, in_dim, hidden_dim, out_dim, name=""):
        """构建嵌入块"""
        layers = []

        layers.append(nn.Linear(in_dim, hidden_dim))

        if self.use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))

        layers.append(nn.ReLU(inplace=False))

        layers.append(nn.Linear(hidden_dim, out_dim))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (..., 14)
        Returns:
            embedded features with the same shape as input
        """
        original_shape = x.shape

        if original_shape[-1] != 14:
            raise ValueError(f"SkeletonFeatureEmbedding期望14维输入，但得到{original_shape[-1]}维")

        x_flat = x.reshape(-1, 14)

        gait_feat = x_flat[:, :3]  # gait
        move_feat = x_flat[:, 3:11]  # move
        pos_feat = x_flat[:, 11:12]  # pos
        angle_feat = x_flat[:, 12:]  # angle

        gait_embedded = self.gait_embed(gait_feat)
        move_embedded = self.move_embed(move_feat)
        pos_embedded = self.pos_embed(pos_feat)
        angle_embedded = self.angle_embed(angle_feat)

        embedded = torch.cat([
            gait_embedded,  # 3
            move_embedded,  # 8
            pos_embedded,  # 1
            angle_embedded  # 2
        ], dim=-1)


        embedded = embedded.view(*original_shape)

        return x + self.residual_scale * embedded




class MaskedBatchNorm2d(nn.BatchNorm2d):
    """
    1. During training, only valid frames are used to compute statistics (including running-stats)
    2. Use detach() to cut off gradients from the mask, avoiding "gradient paths with masks" that cause running-stats drift
    3. During inference, valid-frame statistics are also used, without global averaging
    """

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None):
        if lengths is None:

            return super().forward(x)

        N, C, T, V = x.shape
        device = x.device

        #   (N, T) -> (N, C, T, V)
        mask = create_length_mask(x, lengths)                     # (N, T)
        mask_exp = mask.unsqueeze(1).unsqueeze(-1).expand_as(x)   # (N, C, T, V)

        mask_exp = mask_exp.detach()

        valid_count = mask_exp.sum()

        x_masked = x * mask_exp
        mean = x_masked.sum(dim=(0, 2, 3), keepdim=True) / (valid_count + 1e-8)
        var, _ = torch.var_mean(x_masked, dim=(0, 2, 3), unbiased=False, keepdim=True)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_norm = x_norm * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)

        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze().detach()
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var.squeeze().detach()

        return x_norm

class MaskedBatchNorm1d(nn.BatchNorm1d):
    """
    1. During training, only valid positions are used to compute statistics (including running-stats)
    2. Use detach() to cut off gradients from the mask, avoiding "gradient paths with masks" that cause running-stats drift
    3. During inference, valid-position statistics are also used, without global averaging
    """


    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None):
        if lengths is None:

            return super().forward(x)

        N, C, T = x.shape
        device = x.device

        # (N, T) -> (N, C, T)
        mask = create_length_mask(x, lengths)                 # (N, T)
        mask_exp = mask.unsqueeze(1).expand_as(x)             # (N, C, T)

        mask_exp = mask_exp.detach()

        valid_count = mask_exp.sum()

        x_masked = x * mask_exp
        mean = x_masked.sum(dim=(0, 2), keepdim=True) / (valid_count + 1e-8)
        var, _ = torch.var_mean(x_masked, dim=(0, 2), unbiased=False, keepdim=True)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_norm = x_norm * self.weight.view(1, C, 1) + self.bias.view(1, C, 1)

        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze().detach()
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var.squeeze().detach()

        return x_norm





class MaskedGroupNorm1d(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        assert num_channels % num_groups == 0, \
            f"num_channels({num_channels}) must be divisible by num_groups({num_groups})"
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, lengths=None):
        N, C, L = x.shape
        device = x.device

        # 1. mask
        if lengths is not None:
            mask = create_length_mask(x, lengths)  # (N, L)
            mask = mask.unsqueeze(1).expand(N, C, L).float().detach()
            x_masked = x * mask
        else:
            x_masked = x
            mask = torch.ones(N, C, L, device=device)

        # reshape -> (N, G, C//G, L)
        x_grouped = x_masked.view(N, self.num_groups, C // self.num_groups, L)
        mask_grouped = mask.view(N, self.num_groups, C // self.num_groups, L)

        valid_count = mask_grouped.sum(dim=[2, 3], keepdim=True).clamp(min=1)

        mean = (x_grouped * mask_grouped).sum(dim=[2, 3], keepdim=True) / valid_count

        var = ((x_grouped - mean) ** 2 * mask_grouped).sum(dim=[2, 3], keepdim=True) / valid_count
        var = var.clamp(min=self.eps)  # <-- 关键：防止除 0

        x_norm = (x_grouped - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(N, C, L)

        if self.affine:
            x_norm = x_norm * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)

        if lengths is not None:
            x_norm = x_norm * mask

        return x_norm


class MaskedGroupNorm2d(nn.Module):
    """
    Input: x    (N, C, T, V)
           lengths (N,)  effective frame length for each sample
    Output: same shape as x, padded regions do not participate in statistics, and the output is set to zero again
    """

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        assert num_channels % num_groups == 0, \
            f"num_channels({num_channels}) must be divisible by num_groups({num_groups})"
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, lengths=None):
        N, C, T, V = x.shape
        device = x.device

        if lengths is not None:

            mask = create_length_mask(x, lengths)
            mask = mask.unsqueeze(1).unsqueeze(-1).expand_as(x).float().detach()

            x_masked = x * mask
        else:
            x_masked = x
            mask = torch.ones(N, 1, T, V, device=device)

        # reshape -> (N, G, C//G, T, V)
        x_grouped = x_masked.view(N, self.num_groups, C // self.num_groups, T, V)
        mask_c = mask.expand(N, C, T, V)
        mask_grouped = mask_c.view(N, self.num_groups, C // self.num_groups, T, V)

        mask_grouped = mask_grouped.clamp(min=0)
        valid_count = mask_grouped.sum(dim=[2, 3, 4], keepdim=True).clamp(min=1)

        mean = (x_grouped * mask_grouped).sum(dim=[2, 3, 4], keepdim=True) / valid_count
        var = ((x_grouped - mean) ** 2 * mask_grouped).sum(dim=[2, 3, 4], keepdim=True) / valid_count
        var = var.clamp(min=self.eps)

        x_norm = (x_grouped - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(N, C, T, V)

        if self.affine:
            x_norm = x_norm * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)

        if lengths is not None:
            x_norm = x_norm * mask

        return x_norm


class MaskAwareGlobalPooling(nn.Module):
    def __init__(self, keep_spatial=True):
        super().__init__()
        self.keep_spatial = keep_spatial

    def forward(self, x, lengths=None):
        if lengths is not None:
            N, C, T, V = x.shape
            mask = create_length_mask(x, lengths)
            mask_expanded = mask.unsqueeze(1).unsqueeze(-1).expand(N, C, T, V)
            x_sum = (x * mask_expanded.float()).sum(dim=2)
            valid_counts = mask_expanded.sum(dim=2)         # (N, C, V)
            x_pooled = x_sum / torch.clamp(valid_counts, min=1.0)
            if not self.keep_spatial:
                x_pooled = x_pooled.mean(dim=2, keepdim=True)  # (N, C, 1)
                return x_pooled.unsqueeze(-1).unsqueeze(-1)    # (N, C, 1, 1)
            else:
                return x_pooled.unsqueeze(2)  # (N, C, 1, V)
        else:
            if not self.keep_spatial:
                return F.avg_pool2d(x, x.size()[2:])
            else:
                return F.avg_pool2d(x, (x.size(2), 1))



class MaskAwareChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super().__init__()
        self.reduction = reduction
        self.gn_mask = MaskedGroupNorm2d(num_groups=1, num_channels=in_channels)

        mid = max(in_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self._att_buffer = None

    def forward(self, x, lengths=None):
        N, C, T, V = x.shape
        device = x.device

        if lengths is not None:
            x = self.gn_mask(x, lengths)

        if lengths is not None:
            mask = create_length_mask(x, lengths)              # (N,T)
            mask4d = mask.unsqueeze(1).unsqueeze(-1).float()   # (N,1,T,1)

            x_nan = x.clone()
            x_nan[mask4d.expand_as(x) == 0] = float('nan')
        else:
            x_nan = x
            mask4d = torch.ones((N, 1, T, 1), device=device)

        y = torch.nanmean(x_nan, dim=(2, 3), keepdim=True)   # (N,C,1,1)
        y = torch.nan_to_num(y, nan=0.0)
        att = self.sigmoid(self.fc(y))                       # (N,C,1,1)

        self._att_buffer = att.detach().cpu()

        att = att.expand_as(x)                               # (N,C,T,V)
        att = att * mask4d.expand_as(att)


        return x * att



class MaskAwareTemporalAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()

        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths=None):
        N, C, T, V = x.shape
        device = x.device

        if lengths is not None:
            # (N, T) -> (N, 1, T, 1)
            mask = create_length_mask(x, lengths)
            mask4d = mask.unsqueeze(1).unsqueeze(-1).float()  # (N, 1, T, 1)

            x_masked = x * mask4d

            valid_count = mask4d.sum(dim=(1, 3), keepdim=True).clamp(min=1)  # (N,1,T,1)

            y = x_masked.sum(dim=(1, 3), keepdim=True) / valid_count  # (N,1,T,1)
        else:
            y = x.mean(dim=(1, 3), keepdim=True)  # (N,1,T,1)
            mask4d = torch.ones(N, 1, T, 1, device=device)

        y = y.squeeze(-1)  # (N, 1, T)

        att = self.sigmoid(self.conv(y))  # (N, 1, T)

        att = att.unsqueeze(-1).expand_as(x)  # (N, C, T, V)
        att = att * mask4d.expand_as(att)

        return x * att




class MaskAwareSTGCN(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 reduction=4,
                 has_ca=False,
                 has_ta=True,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)


        if out_channels >= 256:
            effective_gn_divisor = 32
        elif out_channels >= 128:
            effective_gn_divisor = 32
        else:
            effective_gn_divisor = 16

        self.has_ca = has_ca
        self.has_ta = has_ta

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,kernel_size[1])

        self.gcn_module = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.ma_gcn = EfficientValidFrameGCN(self.gcn_module, dropout)



        if self.has_ca:
            self.ca = MaskAwareChannelAttention(in_channels, reduction)


        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=False),
        )
        
        #self.ma_tcn = MaskAwareTCN(out_channels, out_channels, kernel_size[0], stride, dropout, gn_divisor=effective_gn_divisor)

        self.ma_tcn_ms = MaskAwareTCN_MS(out_channels, out_channels, kernel_sizes=[3, 7, 11],
            stride=stride, dropout=0.1, dropout_out=dropout, gn_divisor=effective_gn_divisor)

        if not residual:
            self.residual = lambda x, lengths=None: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = MaskAwareIdentity()
        else:
            # 使用支持mask的残差连接
            self.residual = MaskAwareResidual(in_channels, out_channels, stride, gn_divisor=effective_gn_divisor)


        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, A, lengths):


        if lengths is not None:
            res = self.residual(x, lengths)
        else:
            res = self.residual(x)


        if lengths is not None:
            if self.has_ca:
                x = self.ca(x, lengths) + x
            x_gcn, A = self.ma_gcn(x, A, lengths)
            x_main, new_lengths = self.ma_tcn_ms(x_gcn, lengths)
        else:
            x_gcn, A = self.gcn(x, A)
            x_main = self.tcn(x_gcn)
            new_lengths = None

        x = x_main + res

        return x, A, new_lengths



class MaskAwareIdentity(nn.Module):
    def forward(self, x, lengths=None):
        if lengths is not None:
            mask = create_and_expand_mask(x, lengths)
            return x * mask.float()
        return x


class MaskAwareResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, gn_divisor=32):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=(stride, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        #self.gn = nn.GroupNorm(num_groups=out_channels//16, num_channels=out_channels)
        self.gn_mask = MaskedGroupNorm2d(num_groups=out_channels//gn_divisor, num_channels=out_channels)
    def forward(self, x, lengths=None):
        original_x = x
        N, C_orig, T_orig, V_orig = x.shape

        x = self.conv(x)  # 输出: (N, out_channels, T_new, V)
        N_out, C_out, T_new, V_out = x.shape


        if lengths is not None:

            adjusted_lengths = compute_adjusted_lengths(lengths, self.stride, T_orig, T_new, x.device)
            mask_expanded = create_and_expand_mask(x, adjusted_lengths)
            x = x * mask_expanded.float()
            x = self.gn_mask(x, adjusted_lengths)
        else:
            x = self.bn(x)

        return x


class EfficientValidFrameGCN(nn.Module):
    def __init__(self, gcn_module, dropout):
        super().__init__()
        self.gcn = gcn_module
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, A, lengths=None):
        """
        x: (N, C, T, V)
        A: (K, V, V)
        lengths: (N,)  Valid frame length
        """
        if lengths is None:
            return self.gcn(x, A)

        N, C, T, V = x.shape
        device = x.device

        valid_frames = []
        frame_indices = []
        valid_lengths = []

        for i in range(N):
            if lengths.dim() == 0:
                valid_length = min(int(lengths.item()), T)
            else:
                valid_length = min(int(lengths[i]), T)
            valid_lengths.append(valid_length)

            if valid_length > 0:
                # 提取该样本的所有有效帧
                sample_frames = x[i, :, :valid_length, :]  # (C, valid_T, V)
                valid_frames.append(sample_frames)
                frame_indices.extend([i] * valid_length)

        if not valid_frames:
            return torch.zeros_like(x), A

        all_valid_frames = torch.cat(valid_frames, dim=1)  # (C, total_valid_T, V)
        all_valid_frames = all_valid_frames.unsqueeze(0)  # (1, C, total_valid_T, V)

        valid_output, _ = self.gcn(all_valid_frames, A)  # (1, C, total_valid_T, V)
        valid_output = valid_output.squeeze(0)  # (C, total_valid_T, V)

        #if self.dropout is not None and self.training:
        #    valid_output = self.dropout(valid_output)

        C_out = valid_output.shape[0]

        output = torch.zeros(N, C_out, T, V, device=device, dtype=x.dtype)
        start_idx = 0

        for i in range(N):
            valid_length = valid_lengths[i]
            if valid_length > 0:
                end_idx = start_idx + valid_length
                sample_output = valid_output[:, start_idx:end_idx, :]  # (C, valid_T, V)
                output[i, :, :valid_length, :] = sample_output
                start_idx = end_idx

        return output, A

class MaskAwareTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, gn_divisor=32):
        super().__init__()
        self.stride = stride
        padding = ((kernel_size - 1) // 2, 0)

        self.tcn_layers = nn.ModuleList([
            #MaskedBatchNorm2d(in_channels),
            MaskedGroupNorm2d(num_groups=out_channels//gn_divisor, num_channels=out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout, inplace=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                (kernel_size, 1),
                (stride, 1),
                padding,
            ),
            #MaskedBatchNorm2d(out_channels),
            MaskedGroupNorm2d(num_groups=out_channels//gn_divisor, num_channels=out_channels),
            nn.ReLU(inplace=False),
        ])

    def forward(self, x, lengths=None):
        N, C, T_orig, V = x.shape
        device = x.device

        x = self.tcn_layers[0](x, lengths)  # Norm2d
        x = self.tcn_layers[1](x)  # ReLU

        x = self.tcn_layers[2](x) # Dropout

        x = self.tcn_layers[3](x)  # Conv2d
        T_after_conv = x.shape[3]


        if lengths is not None:
            adjusted_lengths_conv = compute_adjusted_lengths(
                lengths, self.stride, T_orig, T_after_conv, device
            )

            mask_expanded = create_and_expand_mask(x, adjusted_lengths_conv)
            x = x * mask_expanded.float()
        else:
            adjusted_lengths_conv = None

        x = self.tcn_layers[4](x, adjusted_lengths_conv)  # Norm2d
        x = self.tcn_layers[5](x)  # ReLU

        return x, adjusted_lengths_conv


class MaskAwareTCN_MS(nn.Module):
    """
    Multi-scale temporal convolution module
    Kernel sizes: [3, 7, 11]  Intra-branch Dropout=0.1  Fusion Dropout=0.3
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=None, stride=1,
                 dropout=0.1, dropout_out=0.3, gn_divisor=32):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 7, 11]
        self.kernel_sizes = kernel_sizes
        self.num_branches = len(kernel_sizes)
        self.stride = stride

        self.norm_in = MaskedGroupNorm2d(num_groups=out_channels // gn_divisor, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout_in = nn.Dropout(dropout, inplace=False)

        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            padding = ((ks - 1) // 2, 0)
            conv = nn.Conv2d(
                in_channels, out_channels,
                (ks, 1), (stride, 1), padding,
                groups=out_channels, bias=False
            )
            self.branches.append(conv)

        self.branch_dropouts = nn.ModuleList(
            [nn.Dropout(dropout, inplace=False) for _ in range(self.num_branches)]
        )

        self.length_fc = nn.Sequential(
            nn.Linear(1, self.num_branches * 4),
            nn.ReLU(inplace=False),
            nn.Linear(self.num_branches * 4, self.num_branches),
        )

        self.norm_out = MaskedGroupNorm2d(num_groups=out_channels // gn_divisor, num_channels=out_channels)
        self.relu_out = nn.ReLU(inplace=False)
        self.dropout_out = nn.Dropout(dropout_out, inplace=False)

    def forward(self, x, lengths=None):
        N, C, T_orig, V = x.shape
        device = x.device

        x = self.norm_in(x, lengths)
        x = self.relu(x)
        x = self.dropout_in(x)

        branch_outs = []
        for i, (branch, drop) in enumerate(zip(self.branches, self.branch_dropouts)):
            out = branch(x)
            out = drop(out)
            branch_outs.append(out)

        T_new = branch_outs[0].shape[2]
        if lengths is not None and self.stride > 1:
            adjusted_lengths = compute_adjusted_lengths(lengths, self.stride, T_orig, T_new, device)
        else:
            adjusted_lengths = lengths

        if lengths is not None:
            len_feat = (lengths.float() / T_orig).unsqueeze(1)
            len_feat = len_feat.to(device)
        else:
            len_feat = torch.ones(N, 1, device=device) * 0.8

        logits = self.length_fc(len_feat)
        weights = torch.softmax(logits, dim=-1)

        stacked = torch.stack(branch_outs, dim=1)               # (N, B, C, T, V)
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_fused = (stacked * weights).sum(dim=1)

        x_fused = self.norm_out(x_fused, adjusted_lengths)
        x_fused = self.relu_out(x_fused)
        x_fused = self.dropout_out(x_fused)

        return x_fused, adjusted_lengths