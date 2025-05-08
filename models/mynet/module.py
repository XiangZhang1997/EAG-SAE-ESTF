import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from torch.autograd import Variable
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.layers.helpers import to_2tuple

class CoronaryIdentificationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CoronaryIdentificationModule, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels

        self.high_freq_path = nn.Sequential(
            DoubleConv(self.in_channels, self.in_channels)

        )
        self.low_freq_path = nn.Sequential(
            ConvolutionalSelfAttention(self.in_channels)
        )
        self.fusion = EntropyBasedFusion(self.in_channels, self.out_channels)

    def forward(self, x):
        high_freq = self.high_freq_path(x)
        low_freq = self.low_freq_path(x)
        fused = self.fusion(high_freq, low_freq)
        return fused

class ConvolutionalSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(ConvolutionalSelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query_conv(x)
        key = self.key_conv(x)
        attention = query*key
        attention = F.softmax(attention, dim=1)
        value = self.value_conv(x)
        out = value*attention
        return out + x

class EntropyBasedFusion(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def forward(self, high_freq_output, low_freq_output):
        norm_high = torch.sigmoid(high_freq_output)
        norm_low = torch.sigmoid(low_freq_output)
        entropy_high = -torch.sum(norm_high * torch.log(norm_high + 1e-8), dim=1, keepdim=True) 
        entropy_low = -torch.sum(norm_low * torch.log(norm_low + 1e-8), dim=1, keepdim=True) 

        w_h = 1-entropy_high 
        w_l = 1-entropy_low 
        w_total = w_h+w_l 
        weight_high = w_h / (w_total + 1e-8) 
        weight_low = w_l / (w_total + 1e-8) 


        fused_output = weight_high*high_freq_output+weight_low*low_freq_output+high_freq_output+low_freq_output
        return self.conv(fused_output)


class MultiScaleConvattModule(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinearity=nn.ReLU(inplace=True)):
        super(MultiScaleConvattModule, self).__init__()
        self.nonlinearity = nonlinearity

        self.asym1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.asym2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))


        self.sym = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


        self.dilate1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.dilate2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.dilate3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=5, dilation=5)

        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.ca = ChannelAttentionafn(in_channels)
        self.sa = SpatialAttentionafn(in_channels)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x

        asym1_out = self.nonlinearity(self.asym1(x))
        asym2_out = self.nonlinearity(self.asym2(asym1_out))

        sym_out = self.nonlinearity(self.sym(asym2_out))

        dilate1_out = self.nonlinearity(self.dilate1(sym_out))
        dilate2_out = self.nonlinearity(self.conv1x1(self.dilate2(sym_out)))
        dilate3_out = self.nonlinearity(self.conv1x1(self.dilate3(sym_out)))
        dilate4_out = self.nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(sym_out)))))

        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out

        return out

class Attention(nn.Module):
    def __init__(self, dim,H, W, sa_num_heads=4, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., expand_ratio=2):
        super().__init__()

        self.dim = dim
        self.sa_num_heads = sa_num_heads
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."
        self.bn = nn.BatchNorm2d(dim*expand_ratio)
        head_dim = dim // sa_num_heads   # group
        self.scale = qk_scale or (1+1e-6) / (math.sqrt(head_dim)+1e-6) 
        self.q_sgcn = S_GCN(dim,H, W) # replace to gcn
        self.attn_drop = nn.Dropout(attn_drop)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
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
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x_q = x.view(B, H, W, C).permute(0, 3, 1, 2)
        q_sgcn = self.q_sgcn(x_q).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)  
        q_gcn = q_sgcn
        kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q_gcn @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)


        return x

class TransBlock(nn.Module):

    def __init__(self, dim, H, W, sa_num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                    use_layerscale=False, layerscale_value=1e-4, drop=0., attn_drop=0.,
                    drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, expand_ratio=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, H, W, sa_num_heads=sa_num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, expand_ratio=expand_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

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
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x
class OverlapPatchEmbed(nn.Module):

    def __init__(self, patch_size=3, stride=2, in_chans=1, embed_dim=16):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2),bias=False) #
        self.norm = nn.LayerNorm(embed_dim)
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
        x = self.proj(x) 
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) 
        x = self.norm(x)
        return x, H, W

class S_GCN(nn.Module):
    def __init__(self, channel, H,W):
        super(S_GCN, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.para = torch.nn.Parameter(torch.ones((1, channel, H,W), dtype=torch.float32)) 
        self.adj = torch.nn.Parameter(
            torch.ones((H**2,W**2), dtype=torch.float32))  

    def forward(self, x):
        b, c, H, W = x.size()
        n = H * W
        fea_matrix = x.view(b, c, H * W) 
        c_adj = torch.mean(fea_matrix, dim=1) 
        m = torch.ones((b, c, H, W), dtype=torch.float32)
        for i in range(0, b):
            t1 = c_adj[i].unsqueeze(0) 
            t2 = t1.t()  

            c_adj_ = torch.abs(torch.abs(torch.sigmoid(t1 - t2) - 0.5) - 0.5) * 2 
            c_adj_s = (c_adj_.t() + c_adj_) / 2
            output0 = torch.mul(torch.mm(fea_matrix[i], self.adj * c_adj_s).view(1, c, H, W),  
                                self.para)  
            m[i] = output0

        output = torch.nn.functional.relu(m.cuda()) 

        print("output.shape",output.shape)
        print(t1.shape, (t1 - t2).shape,c_adj_.shape,c_adj_s.shape)
        print(fea_matrix.shape,(self.adj * c_adj_s).shape)
        return output

class LongDistanceDependencyModule_onlytrans(nn.Module):
    def __init__(self, in_channels, H, W, num_heads=4, hidden_dim=256):
        super(LongDistanceDependencyModule_onlytrans, self).__init__()
        self.transformer = TransBlock(dim=in_channels, H=H, W=W)

        self.s_gcn = S_GCN(in_channels, H, W)

        self.fusion = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        x_trans = self.transformer(x_seq)
        x_trans = x_trans.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)


        x_gcn = self.s_gcn(x)

        x_fused = torch.cat([x_trans, x_gcn], dim=1)  # (B, 2*C, H, W)
        x_out = self.fusion(x_fused)  # (B, C, H, W)
        return x_out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size,kernel_size), padding=padding),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size,kernel_size), padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,k=2,s=2, bilinear=False): 
        super().__init__()

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=k, mode='bilinear', align_corners=True),
                DoubleConv(in_channels, out_channels)
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k,padding=0, stride=s, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)

            )

    def forward(self, x):
        x = self.up(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))

    def forward(self, x):
        return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
