import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from networks.bra_block import Block
from einops import rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from networks.bra_decoder_expandx4 import BasicLayer_up
from networks.DA_block import DANetHead

class SCCSA(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(SCCSA, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),~
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        return out

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)
    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = x.permute(0,2,3,1)
        x = self.expand(x)
        B, H, W, C = x.shape
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        return x

from torch.nn import Softmax


# 定义一个无限小的矩阵，用于在注意力矩阵中屏蔽特定位置
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        # Q, K, V转换层
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # 使用softmax对注意力分数进行归一化
        self.softmax = Softmax(dim=3)
        self.INF = INF
        # 学习一个缩放参数，用于调节注意力的影响
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        # 计算查询(Q)、键(K)、值(V)矩阵
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)

        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        # 计算垂直和水平方向上的注意力分数，并应用无穷小掩码屏蔽自注意
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width).to('cuda')).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)

        # 在垂直和水平方向上应用softmax归一化
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        # 分离垂直和水平方向上的注意力，应用到值(V)矩阵上
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        # 计算最终的输出，加上输入x以应用残差连接
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        return self.gamma * (out_H + out_W) + x

from torch.nn import init
from torch.nn import functional as F


class DoubleAttention(nn.Module):

    def __init__(self, in_channels, c_m, c_n, reconstruct=True):
        super().__init__()
        self.in_channels = in_channels  # 输入通道数
        self.reconstruct = reconstruct  # 是否需要重构输出以匹配输入的维度
        self.c_m = c_m  # 第一个注意力机制的输出通道数
        self.c_n = c_n  # 第二个注意力机制的输出通道数
        # 定义三个1x1卷积层，用于生成A、B和V特征
        self.convA = nn.Conv2d(in_channels, c_m, 1)
        self.convB = nn.Conv2d(in_channels, c_n, 1)
        self.convV = nn.Conv2d(in_channels, c_n, 1)
        # 如果需要重构，定义一个1x1卷积层用于输出重构
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # 前向传播
        b, c, h, w = x.shape
        assert c == self.in_channels  # 确保输入通道数与初始化时一致
        A = self.convA(x)  # b,c_m,h,w# 生成A特征图
        B = self.convB(x)  # b,c_n,h,w# 生成B特征图
        V = self.convV(x)  # b,c_n,h,w# 生成V特征图
        # 将特征图维度调整为方便矩阵乘法的形状
        tmpA = A.view(b, self.c_m, -1)
        attention_maps = F.softmax(B.view(b, self.c_n, -1),dim=-1)
        attention_vectors = F.softmax(V.view(b, self.c_n, -1),dim=-1)
        # 步骤1: 特征门控
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # b.c_m,c_n
        # 步骤2: 特征分配
        tmpZ = global_descriptors.matmul(attention_vectors)  # b,c_m,h*w
        tmpZ = tmpZ.view(b, self.c_m, h, w)  # b,c_m,h,w
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)  # 如果需要，通过重构层调整输出通道数

        return tmpZ
# block = DoubleAttention(64, 128, 128)
# input = torch.rand(1, 64, 64, 64)
# output = block(input)
# print(input.size(), output.size())


    
class BRCCAUnetSystem(nn.Module):
    def __init__(self, img_size=256,
                 depth=[2, 2, 8, 2],
                 depths_decoder=[2, 8, 2, 2], 
                 in_chans=3, num_classes=1000, 
                 embed_dim=[96, 192, 384, 768],
                 head_dim=32, qk_scale=None, representation_size=None,
                 drop_path_rate=0.,
                 use_checkpoint_stages=[],
                 norm_layer=nn.LayerNorm,
                 ########
                 n_win=7,
                 kv_downsample_mode='identity',
                 kv_per_wins=[2, 2, -1, -1],
                 topks=[2, 4, 8, -2],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[96, 192, 384, 768],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=False,
                 #-----------------------
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1], # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[3, 3, 3, 3],
                 param_attention='qkvo',
                 final_upsample = "expand_first",
                 mlp_dwconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim[0]  # num_features for consistency with other models
        patches_resolution = [img_size // 4, img_size // 4]
        self.num_layers = len(depth)
        self.patches_resolution = patches_resolution
        self.final_upsample = final_upsample

        
        self.sccsa1 = CrissCrossAttention(embed_dim[1])
        self.sccsa2 = CrissCrossAttention(embed_dim[2])
        self.sccsa3 = CrissCrossAttention(embed_dim[3])
        
        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i+1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i+1])
            )
            self.downsample_layers.append(downsample_layer)
        ##########################################################################

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i],
                        input_resolution=(patches_resolution[0] // (2 ** i),
                                          patches_resolution[1] // (2 ** i)),
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        topk=topks[i],
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        qk_scale=qk_scale,
                        kv_per_win=kv_per_wins[i],
                        kv_downsample_ratio=kv_downsample_ratios[i],
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,
                        param_attention=param_attention,
                        param_routing=param_routing,
                        diff_routing=diff_routing,
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]
            
            
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*embed_dim[self.num_layers - 1 - i_layer],
                                      embed_dim[self.num_layers - 1 - i_layer]) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=embed_dim[self.num_layers - 1 - i_layer], dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(
                    dim=embed_dim[self.num_layers - 1 - i_layer],
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    depth=depths_decoder[i_layer],
                    embed_dim=embed_dim [self.num_layers - 1 - i_layer],
                    num_heads=nheads[(self.num_layers - 1 - i_layer)],
                    drop_path_rate=drop_path_rate,
                    layer_scale_init_value=-1,
                    topks=topks[3 - i_layer],
                    qk_dims=qk_dims[3 - i_layer],
                    n_win=n_win,
                    kv_per_wins=kv_per_wins[3 - i_layer],
                    kv_downsample_kernels=[3 - i_layer],
                    kv_downsample_ratios=[3 - i_layer],
                    kv_downsample_mode=kv_downsample_mode,
                    param_attention=param_attention,
                    param_routing=param_routing,
                    diff_routing=diff_routing,
                    soft_routing=soft_routing,
                    pre_norm=pre_norm,
                    mlp_ratios=mlp_ratios[3 - i_layer],
                    mlp_dwconv=mlp_dwconv,
                    side_dwconv=side_dwconv,
                    qk_scale=qk_scale,
                    before_attn_dwconv=before_attn_dwconv,
                    auto_pad=auto_pad,
                    norm_layer=nn.LayerNorm,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
        self.norm_up = norm_layer(embed_dim[0])
        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up4 = FinalPatchExpand_X4(input_resolution=(img_size // 4, img_size // 4),
                                          dim_scale=4, dim=embed_dim[0])
        self.output = nn.Conv2d(in_channels=embed_dim[0], out_channels=self.num_classes, kernel_size=1, bias=False)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x_downsample = []
        for i in range(3):
            x = self.downsample_layers[i](x) # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
            x = x.flatten(2).transpose(1, 2)
            print('q',x.shape)
            
            x = self.stages[i](x)
            
            
            print('h',x.shape)
            x_downsample.append(x)
            B, L, C = x.shape
            x = x.reshape(B,int(math.sqrt(L)),int(math.sqrt(L)),C)
            x = x.permute(0,3,1,2)
        x = self.downsample_layers[3](x)
        return x, x_downsample
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            elif inx == 1:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                B, L, C = x.shape
                x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C)
                x = x.permute(0, 3, 1, 2)
                x = self.sccsa3(x)
                x = x.flatten(2).transpose(1, 2)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
            elif inx == 2:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                B, L, C = x.shape
                x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C)
                x = x.permute(0, 3, 1, 2)
                x = self.sccsa2(x)
                x = x.flatten(2).transpose(1, 2)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                B, L, C = x.shape
                x = x.view(B, int(math.sqrt(L)), int(math.sqrt(L)), C)
                x = x.permute(0, 3, 1, 2)
                x = self.sccsa1(x)
                x = x.flatten(2).transpose(1, 2)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        x = self.norm_up(x)  # B L C
        return x
    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"
        if self.final_upsample == "expand_first":
            x = self.up4(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)
        return x
    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        return x
    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.stages):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
