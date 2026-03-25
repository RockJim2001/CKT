from mmcv.runner import BaseModule, auto_fp16
from mmdet.models.builder import NECKS
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmfewshot.detection.models.utils.dilateformer import DilateAttention
import torch


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 16, bias=False),
            nn.ReLU(),
            nn.Linear(channel // 16, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1,16,64,64
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # 1,16
        # print(y.size())
        y = self.fc(y).view(b, c, 1, 1)
        # 1,16,1,1
        # print(y.size())
        # print(y.expand_as(x))
        # y.expand_as(x) 把y变成和x一样的形状
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), x.size(1)))
        max_out = self.fc(self.max_pool(x).view(x.size(0), x.size(1)))
        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(2).unsqueeze(3)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class AdaptiveDCA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilations=(1, 2, 4, 8)):
        super(AdaptiveDCA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_dilation = len(dilations)
        assert num_heads % self.num_dilation == 0, \
            f"num_heads {num_heads} must be a multiple of num_dilation {self.num_dilation}!"

        self.heads_per_branch = num_heads // self.num_dilation
        self.branch_dim = dim // self.num_dilation
        self.head_dim = self.branch_dim // self.heads_per_branch  # ⚠️ 修正

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)

        self.branches = nn.ModuleList([
            DilateAttention(self.head_dim, qk_scale, attn_drop, kernel_size, d)
            for d in dilations
        ])

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, self.num_dilation, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, self.branch_dim, H, W)
        qkv = qkv.permute(2, 1, 0, 3, 4, 5).contiguous()

        outs = []
        for i in range(self.num_dilation):
            q, k, v = qkv[i][0], qkv[i][1], qkv[i][2]
            out = self.branches[i](q, k, v)  # [B, H, W, branch_dim]
            out = out.permute(0, 3, 1, 2).contiguous()
            outs.append(out)

        weights = self.attn(x)  # [B, num_dilation, 1, 1]
        weights = F.dropout(weights, p=0.1, training=self.training)

        out_chunks = []
        for i in range(self.num_dilation):
            out_chunks.append(outs[i] * weights[:, i:i + 1, :, :])

        out = torch.cat(out_chunks, dim=1)  # 沿 channel 拼
        return out


class AdaptiveResidualBlock(nn.Module):
    def __init__(self, out_channels, dilations):
        super().__init__()
        # self.adaptive = AdaptiveDCA(out_channels, dilations=dilations)
        self.adaptive = AdaptiveDCA(out_channels)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.conv1x1(self.adaptive(x))
        weight = self.gate(res)
        return x + res * weight


class LocalWindowDynamicDCA(nn.Module):
    def __init__(self, in_channels, window_size=7, dilations=[1, 2]):
        super(LocalWindowDynamicDCA, self).__init__()
        self.window_size = window_size
        self.dilations = dilations

        self.branches = nn.ModuleList([
                                          nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
                                      ] + [
                                          nn.Conv2d(in_channels, in_channels, 3, padding=d, dilation=d,
                                                    groups=in_channels)
                                          for d in dilations
                                      ])

        self.num_branches = len(self.branches)

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, self.num_branches, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        stride = self.window_size // 2  # overlap 50%

        pad_h = (stride - (H - self.window_size) % stride) % stride
        pad_w = (stride - (W - self.window_size) % stride) % stride

        Hp, Wp = H + pad_h, W + pad_w
        x_padded = F.pad(x, (0, pad_w, 0, pad_h))

        # unfold: [B, C * K * K, L]
        x_unfold = F.unfold(x_padded, kernel_size=self.window_size, stride=stride)
        B, _, L = x_unfold.shape

        # [B*L, C, K, K]
        x_windows = x_unfold.transpose(1, 2).contiguous().view(B * L, C, self.window_size, self.window_size)

        branch_outs = [branch(x_windows) for branch in self.branches]
        branch_outs = torch.stack(branch_outs, dim=1)  # [B*L, num_branches, C, K, K]

        attn = self.attn(x_windows)  # [B*L, num_branches, 1, 1]
        attn = attn.unsqueeze(2)  # [B*L, num_branches, 1, 1, 1]

        out = (branch_outs * attn).sum(dim=1)  # [B*L, C, K, K]

        out = out.view(B, L, C * self.window_size * self.window_size).transpose(1, 2).contiguous()
        out = F.fold(out, output_size=(Hp, Wp), kernel_size=self.window_size, stride=stride)

        # norm map
        norm_map = F.fold(
            F.unfold(torch.ones_like(x_padded), kernel_size=self.window_size, stride=stride),
            output_size=(Hp, Wp), kernel_size=self.window_size, stride=stride)
        out = out / norm_map

        out = out[:, :, :H, :W]
        return out


@NECKS.register_module()
class FPNWithAdaptiveDCA(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 dca_mode='fpn',
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')
                 ):
        super(FPNWithAdaptiveDCA, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.dca_mode = dca_mode
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # 其中
        # dca_mode
        # 可以取：
        #
        # "fpn" → 原始FPN版本
        #dca_mode
        # "adaptive" → 仅用
        # AdaptiveDCA
        #
        # "local" → 仅用
        # LocalWindowDynamicDCA
        #
        # "hybrid" → LocalWindowDynamicDCA + AdaptiveDCA
        for i in range(self.start_level, self.backbone_end_level):
            # 1、原始FPN版本 对应路径为： '/work_dirs/dior/split1-FPN/'
            if self.dca_mode == 'fpn':
                l_conv = ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False)
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
            # 2、使用AdaptiveDCA的版本，对应路径为： '/work_dirs/dior/split1-AdaptiveDCA'
            elif self.dca_mode == 'adaptive':
                l_conv = ConvModule(in_channels[i], out_channels, 1, conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                                    act_cfg=act_cfg,
                                    inplace=False)
                # ⏬ 在 C4 和 C5 后面加入Adaptive
                if i == 1:  # C4上加入AdaptiveDCA
                    l_conv = nn.Sequential(
                        l_conv,
                        # AdaptiveDCA(out_channels, dilations=[1, 2])
                        AdaptiveResidualBlock(out_channels, [1, 2])
                    )
                elif i == 2:  # C5上加入AdaptiveDCA
                    l_conv = nn.Sequential(
                        l_conv,
                        # AdaptiveDCA(out_channels, dilations=[2, 4])
                        AdaptiveResidualBlock(out_channels, [2, 4])
                    )
                elif i == 3:  # C6上加入AdaptiveDCA
                    l_conv = nn.Sequential(
                        l_conv,
                        # AdaptiveDCA(out_channels, dilations=[3, 5])
                        AdaptiveResidualBlock(out_channels, dilations=[3, 5])
                    )
                fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            # 3、仅仅使用LocalWindowDynamicDCA模块的版本 对应路径为： '/work_dirs/dior/split1-LocalWindowDynamicDCA'
            elif self.dca_mode == 'localwindow':
                l_conv = nn.Conv2d(in_channels[i], out_channels, 1)
                if i == 0:
                    # C3: LocalWindowDynamicDCA
                    l_conv = nn.Sequential(
                        l_conv,
                        LocalWindowDynamicDCA(out_channels, window_size=5, dilations=[1, 2])
                    )
                fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            elif self.dca_mode == 'hybrid':
                l_conv = nn.Conv2d(in_channels[i], out_channels, 1)
                if i == 0:
                    # C3: LocalWindowDynamicDCA
                    l_conv = nn.Sequential(
                        l_conv,
                        LocalWindowDynamicDCA(out_channels, window_size=5, dilations=[1, 2])
                    )
                elif i >= self.backbone_end_level - 2:
                    l_conv = nn.Sequential(
                        l_conv,
                        AdaptiveDCA(out_channels)
                    )
                fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            elif self.dca_mode == 'se':
                # 与原始FPN一样的 1×1 conv
                l_conv = ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False
                )
                # 在 lateral conv 后接 SE 模块
                l_conv = nn.Sequential(
                    l_conv,
                    SELayer(out_channels, reduction=16)
                )
                # fpn_conv 同正常FPN
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
            elif self.dca_mode == 'cbam':
                l_conv = ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False
                )
                # 接入 CBAM
                l_conv = nn.Sequential(
                    l_conv,
                    CBAM(out_channels)
                )
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
