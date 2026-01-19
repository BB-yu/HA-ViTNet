import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath
from networks.bra import BiLevelRoutingAttention


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
    def forward(self, x):
        """
        x: NHWC tensor
        """
        x = x.permute(0, 3, 1, 2)  # NCHW
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NHWC
        return x


class Attention(nn.Module):
    """
    vanilla attention
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        args:
            x: NHWC tensor
        return:
            NHWC tensor
        """
        _, H, W, _ = x.size()
        x = rearrange(x, 'n h w c -> n (h w) c')

        #######################################
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #######################################

        x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)
        return x


class AttentionLePE(nn.Module):
    """
    attention with LePE
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

    def forward(self, x):
        """
        args:
            x: NHWC tensor
        return:
            NHWC tensor
        """
        _, H, W, _ = x.size()
        x = rearrange(x, 'n h w c -> n (h w) c')

        #######################################
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        lepe = self.lepe(rearrange(x, 'n (h w) c -> n c h w', h=H, w=W))
        lepe = rearrange(lepe, 'n c h w -> n (h w) c')

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + lepe

        x = self.proj(x)
        x = self.proj_drop(x)
        #######################################

        x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)
        return x


class nchwAttentionLePE(nn.Module):
    """
    Attention with LePE, takes nchw input
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

    def forward(self, x: torch.Tensor):
        """
        args:
            x: NCHW tensor
        return:
            NCHW tensor
        """
        B, C, H, W = x.size()
        q, k, v = self.qkv.forward(x).chunk(3, dim=1)  # B, C, H, W

        attn = q.view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2) @ \
               k.view(B, self.num_heads, self.head_dim, H * W)
        attn = torch.softmax(attn * self.scale, dim=-1)
        attn = self.attn_drop(attn)

        # (B, nhead, HW, HW) @ (B, nhead, HW, head_dim) -> (B, nhead, HW, head_dim)
        output: torch.Tensor = attn @ v.view(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
        output = output.permute(0, 1, 3, 2).reshape(B, C, H, W)
        output = output + self.lepe(v)

        output = self.proj_drop(self.proj(output))
        return output


class Mlp(nn.Module):  # 搭建前馈神经网络
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一层全连接层
        self.act = act_layer()  # 使用nn.GELU作为激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二层全连接层
        self.drop = nn.Dropout(drop)  # 使用概率drop=0.失活

    def forward(self, x):  # 前向传播，输入数据x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class GateMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4., mlp_dwconv=False):
        super().__init__()
        self.proj_1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.dwconv = DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity()
        self.activation = nn.GELU()
        self.proj_2 = nn.Linear(int(mlp_ratio * dim), dim)
        # 新增一个线性层用于门控分支
        self.gate_proj = nn.Linear(int(mlp_ratio * dim), dim)

    def forward(self, x):
        x = self.proj_1(x)
        x = self.dwconv(x)
        x = self.activation(x)
        # 计算门控信号
        gate = self.gate_proj(x)
        # 门控操作，这里简单使用逐元素相乘
        x = self.proj_2(x) * gate
        return x
    
class Block(nn.Module):
    def __init__(self, dim, input_resolution, drop_path=0., layer_scale_init_value=-1,num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=4, mlp_dwconv=False, side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim
        self.input_resolution=input_resolution
        
        # Positional Embedding
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
            
        # Attention modules for both branches
        if topk > 0:
            self.attn_pre = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                                 qk_scale=qk_scale, kv_per_win=kv_per_win,
                                                 kv_downsample_ratio=kv_downsample_ratio,
                                                 kv_downsample_kernel=kv_downsample_kernel,
                                                 kv_downsample_mode=kv_downsample_mode,
                                                 topk=topk, param_attention=param_attention, param_routing=param_routing,
                                                 diff_routing=diff_routing, soft_routing=soft_routing,
                                                 side_dwconv=side_dwconv, auto_pad=auto_pad)
            self.attn_post = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                                 qk_scale=qk_scale, kv_per_win=kv_per_win,
                                                 kv_downsample_ratio=kv_downsample_ratio,
                                                 kv_downsample_kernel=kv_downsample_kernel,
                                                 kv_downsample_mode=kv_downsample_mode,
                                                 topk=topk, param_attention=param_attention, param_routing=param_routing,
                                                 diff_routing=diff_routing, soft_routing=soft_routing,
                                                 side_dwconv=side_dwconv, auto_pad=auto_pad)
        # ... 其他 attn 类型省略以简化 ...

        # Normalization and MLP layers are shared
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = GateMLP(dim, mlp_ratio=mlp_ratio)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # New: learnable weights for fusion
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        # 1. 保存原始输入用于最终残差连接
        shortcut = x
        x_nhwc = x.view(B, H, W, C)
        x_nhwc_permute = x_nhwc.permute(0, 3, 1, 2)
        
        # 2. DWConv Positional Embedding
        x_nhwc_permute = x_nhwc_permute + self.pos_embed(x_nhwc_permute)
        x_nhwc = x_nhwc_permute.permute(0, 2, 3, 1)

        # 3. 双分支注意力计算
        # Pre-Norm branch: norm -> attn
        x_pre_attn = self.attn_pre(self.norm1(x_nhwc))
        
        # Post-Norm branch: attn
        x_post_attn = self.attn_post(x_nhwc)

        # 4. 加权融合注意力输出
        weights = torch.softmax(torch.cat([self.alpha, self.beta], dim=0), dim=0)
        x_fused_attn = weights[0] * x_pre_attn + weights[1] * x_post_attn
        x_fused_attn = self.drop_path(x_fused_attn)
        
        # 5. 内部残差连接和MLP
        x_with_attn = x_nhwc + x_fused_attn
        
        # 6. MLP部分的Pre-Norm (先归一化再残差)
        x_mlp = self.mlp(self.norm2(x_with_attn))
        x_mlp = self.drop_path(x_mlp)
        
        x_out_nhwc = x_with_attn + x_mlp
        
        # 7. 最终残差连接
        final_output = x_out_nhwc.flatten(1, 2)
        
        return final_output
# class Block(nn.Module):
#     def __init__(self, dim, input_resolution, drop_path=0., layer_scale_init_value=-1,num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
#                  kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
#                  topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
#                  mlp_ratio=4, mlp_dwconv=False, side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
#         super().__init__()
#         qk_dim = qk_dim or dim
#         self.input_resolution=input_resolution
#         # modules
#         if before_attn_dwconv > 0:
#             self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
#         else:
#             self.pos_embed = lambda x: 0
#         self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
#         if topk > 0:
#             self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
#                                                 qk_scale=qk_scale, kv_per_win=kv_per_win,
#                                                 kv_downsample_ratio=kv_downsample_ratio,
#                                                 kv_downsample_kernel=kv_downsample_kernel,
#                                                 kv_downsample_mode=kv_downsample_mode,
#                                                 topk=topk, param_attention=param_attention, param_routing=param_routing,
#                                                 diff_routing=diff_routing, soft_routing=soft_routing,
#                                                 side_dwconv=side_dwconv,
#                                                 auto_pad=auto_pad)
#         elif topk == -1:
#             self.attn = Attention(dim=dim)
#         elif topk == -2:
#             self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
#         elif topk == 0:
#             self.attn = nn.Sequential(rearrange('n h w c -> n c h w'),  # compatiability
#                                       nn.Conv2d(dim, dim, 1),  # pseudo qkv linear
#                                       nn.Conv2d(dim, dim, 5, padding=2, groups=dim),  # pseudo attention
#                                       nn.Conv2d(dim, dim, 1),  # pseudo out linear
#                                       rearrange('n c h w -> n h w c')
#                                       )
#         self.norm2 = nn.LayerNorm(dim, eps=1e-6)

#         # self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
#         #                          DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
#         #                          nn.GELU(),
#         #                          nn.Linear(int(mlp_ratio * dim), dim)
#         #                          )

#         self.mlp = GateMLP(dim, mlp_ratio=mlp_ratio) #m门控
        
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         # tricks: layer scale & pre_norm/post_norm
#         if layer_scale_init_value > 0:
#             self.use_layer_scale = True
#             self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
#             self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
#         else:
#             self.use_layer_scale = False
#         self.pre_norm = pre_norm
        
        
#         self.alpha = nn.Parameter(torch.zeros(1))  # 注意力路径权重 
#         self.beta = nn.Parameter(torch.ones(1))    # 主路径权重 
#         self.width_weights = nn.Parameter(torch.ones(1, 1,dim))  # 通道级宽度连接 (
        
#     def forward(self, x):
#         """
#         x: NCHW tensor
#         """
#         H, W = self.input_resolution

#         B, L, C = x.shape

        
#         assert L == H * W, "input feature has wrong size"

#         shortcut = x
#         x = self.norm1(x)
#         x = x.view(B, H, W, C)

#         x = x.permute(0, 3, 1, 2)

#         # conv pos embedding
#         x = x + self.pos_embed(x)
#         # permute to NHWC tensor for attention & mlp
#         x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#         # print("x.shape",x.shape)

#         # attention & mlp
#         # if self.pre_norm:
#         #     if self.use_layer_scale:
#         #         x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
#         #         x = x + self.drop_path(self.gamma2 * self.WF(self.norm2(x)))  # (N, H, W, C)
#         #     else:
#         #         x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, H, W, C)
#         #         x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, H, W, C)
#         # else:  # https://kexue.fm/archives/9009
#         #     if self.use_layer_scale:
#         #         x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
#         #         x = self.norm2(x + self.drop_path(self.gamma2 * self.WF(x)))  # (N, H, W, C)
#         #     else:
#         #         x = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
#         #         x = self.norm2(x + self.drop_path(self.mlp(x)))  # (N, H, W, C)
#         dualattn=True
#         if dualattn:
#             x1 =  self.drop_path(self.attn(self.norm1(x)))  # (N, H, W, C)
#             x1 = self.drop_path(self.mlp(self.norm2(x1)))  # (N, H, W, C)

#             # x1 = x + self.drop_path(self.attn(self.norm1(x))) 
#             # x2 = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
#             # x2= x1+x2
#             # x2 = self.norm2(x2 + self.drop_path(self.mlp(x2)))  # (N, H, W, C)

            

#         # permute back
#         x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
#         x = x.flatten(2).transpose(1, 2)+shortcut
        
#         return x
    
    


