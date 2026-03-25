from functools import partial
import torch.nn.functional as F
import torch
from mmdet.models import DETECTORS, TwoStageDetector
from torch import nn

from mmfewshot.detection.models.utils.dilateformer import DilateBlock


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


class ResidualDilateBlock(nn.Module):
    def __init__(self, dim, num_heads, dilation, kernel_size=3):
        super().__init__()
        self.dilateblock = DilateBlock(
            dim=dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            mlp_ratio=4.,
            qkv_bias=True
        )
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn_out = self.dilateblock(x)
        gate_mask = self.gate(x)
        out = attn_out * gate_mask + x * (1 - gate_mask)
        return out


class LocalWindowDynamicDCA(nn.Module):
    def __init__(self, in_channels, window_size=7, dilations=[1, 2, 3]):
        super(LocalWindowDynamicDCA, self).__init__()
        self.window_size = window_size
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 3, padding=d, dilation=d, groups=in_channels)
            for d in dilations
        ])
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, len(dilations), 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h))

        _, _, Hp, Wp = x.shape

        x_windows = x.unfold(2, self.window_size, self.window_size).unfold(3, self.window_size, self.window_size)
        B, C, num_h, num_w, Wh, Ww = x_windows.shape
        x_windows = x_windows.contiguous().view(-1, C, Wh, Ww)

        branch_outs = [branch(x_windows) for branch in self.branches]
        branch_outs = torch.stack(branch_outs, dim=1)  # [B*num_h*num_w, num_branch, C, H, W]

        attn = self.attn(x_windows)  # [B*num_h*num_w, num_branch, 1, 1]
        attn = attn.unsqueeze(2)  # [B*num_h*num_w, num_branch, 1, 1, 1]

        out = (branch_outs * attn).sum(dim=1)

        out = out.view(B, num_h, num_w, C, Wh, Ww)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, Hp, Wp)
        return out[:, :, :H, :W]


@DETECTORS.register_module()
class GeneralizedFewShotDetector(TwoStageDetector):
    def __init__(self, backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(GeneralizedFewShotDetector, self).__init__(backbone,
                                                         neck,
                                                         rpn_head,
                                                         roi_head,
                                                         train_cfg,
                                                         test_cfg,
                                                         pretrained,
                                                         init_cfg)
        out_channels = [256, 512, 1024, 2048]
        self.dca_modules = nn.ModuleList()
        for i in range(len(out_channels)):
            if i in [0, 1]:  # 对应 C2, C3
                self.dca_modules.append(
                    LocalWindowDynamicDCA(out_channels[i], window_size=7, dilations=[1, 2])
                )
            else:  # C4, C5
                self.dca_modules.append(
                    LocalWindowDynamicDCA(out_channels[i], window_size=7, dilations=[1, 2, 3])
                )

        # 空间-通道注意力机制
        # self.cbam = CBAM(in_channels=256)
        # self.cbam_0 = CBAM(in_channels=256)
        # self.cbam_1 = CBAM(in_channels=512)
        # self.cbam_2 = CBAM(in_channels=1024)
        # self.cbam_3 = CBAM(in_channels=2048)

        # self.se_0 = SELayer(channel=256)
        # self.se_1 = SELayer(channel=512)
        # self.se_2 = SELayer(channel=1024)
        # self.se_3 = SELayer(channel=2048)

        # # num_heads = [2, 4, 8, 16]  # 调参
        # # num_heads = [4, 8, 16, 32]  # 调参
        # num_heads = [8, 16, 32, 64]  # 调参
        # kernel_size = 3  # 调参
        # # dilation = [1, 2]  # 调参
        # # dilation = [1, 2]  # 调参
        # # dilation = [1, 2]  # 调参
        # dilation = [1, 2, 4, 8]  # 调参
        # mlp_ratio = 4.  # 调参
        # # mlp_ratio = 1.  # 调参
        # # mlp_ratio = 2.  # 调参
        # # mlp_ratio = 3.  # 调参
        # qkv_bias = True
        # qk_scale = None
        # drop = 0.
        # attn_drop = 0.
        # drop_path = 0.1
        # norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # cpe_per_block = False
        # dim = [256, 512, 1024, 2048]  # 不需要调参，只需要看看哪一个被使用
        # act_layer = nn.GELU
        # 256, 512, 1024, 2048
        # self.dilateblock_0 = DilateBlock(dim=dim[0],
        #                                  num_heads=num_heads[0],
        #                                  kernel_size=kernel_size, dilation=dilation,
        #                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                                  qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
        #                                  drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
        #                                  norm_layer=norm_layer,
        #                                  act_layer=act_layer,
        #                                  cpe_per_block=cpe_per_block)
        #
        # self.dilateblock_1 = DilateBlock(dim=dim[1], num_heads=num_heads[1],
        #                                  kernel_size=kernel_size, dilation=dilation,
        #                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                                  qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
        #                                  drop_path=drop_path[1] if isinstance(drop_path, list) else drop_path,
        #                                  norm_layer=norm_layer,
        #                                  act_layer=act_layer,
        #                                  cpe_per_block=cpe_per_block)
        #
        # self.dilateblock_2 = DilateBlock(dim=dim[2], num_heads=num_heads[2],
        #                                  kernel_size=kernel_size, dilation=dilation,
        #                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                                  qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
        #                                  drop_path=drop_path[2] if isinstance(drop_path, list) else drop_path,
        #                                  norm_layer=norm_layer,
        #                                  act_layer=act_layer,
        #                                  cpe_per_block=cpe_per_block)

        # self.dilateblock_3 = DilateBlock(dim=dim[3], num_heads=num_heads[3],
        #                                  kernel_size=kernel_size, dilation=dilation,
        #                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #                                  qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
        #                                  drop_path=drop_path[3] if isinstance(drop_path, list) else drop_path,
        #                                  norm_layer=norm_layer,
        #                                  act_layer=act_layer,
        #                                  cpe_per_block=cpe_per_block)

        # self.fpn_attn_c4 = ResidualDilateBlock(dim=512, num_heads=8, dilation=[1, 2])
        # self.fpn_attn_c5 = ResidualDilateBlock(dim=1024, num_heads=8, dilation=[1, 2, 4, 8])
        # self.fpn_attn_c6 = ResidualDilateBlock(dim=2048, num_heads=8, dilation=[2, 4, 8, 16])

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        # # 对其中的x添加注意力机制

        x_list = list(x)

        # x_list[0] = self.se_0(x_list[0])
        # x_list[1] = self.se_1(x_list[1])
        # x_list[2] = self.se_2(x_list[2])
        # x_list[3] = self.se_3(x_list[3])

        # x_list[0] = self.cbam_0(x_list[0])
        # x_list[1] = self.cbam_1(x_list[1])
        # x_list[2] = self.cbam_2(x_list[2])
        # x_list[3] = self.cbam_3(x_list[3])

        # x_list[0] = self.dilateblock_0(x_list[0])
        # x_list[1] = self.dilateblock_1(x_list[1])
        # x_list[2] = self.dilateblock_2(x_list[2])
        # x_list[3] = self.dilateblock_3(x_list[3])
        # x_list[1] = x_list[1]
        # x_list[2] = x_list[2]
        # x_list[3] = x_list[3]

        # # 用 ResidualDilateBlock 改高层
        # x_list = list(x)

        outs_dca = []
        for i, out in enumerate(x_list):
            outs_dca.append(self.dca_modules[i](out))

        # x = tuple(x_list)

        x = tuple(outs_dca)
        if self.with_neck:
            x = self.neck(x)

        return x
