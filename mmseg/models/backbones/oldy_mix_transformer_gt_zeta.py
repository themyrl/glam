# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
import math


from .mix_transformer import mit_b4
from einops import repeat, rearrange


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[0], window_size[1], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class ClassiqueMlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ClassicAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, pe):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        pe = rearrange(pe, 'h w g c -> (h w g) c')
        x = x + pe[None,...]

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Attention(nn.Module):
    def __init__(self, attn, gt_num=1, window_size=(8,8)):
        super().__init__()

        self.dim = attn.dim
        self.num_heads = attn.num_heads
        self.scale = attn.scale

        self.q = attn.q
        self.kv = attn.kv
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

        self.sr_ratio = attn.sr_ratio
        if self.sr_ratio > 1:
            self.sr = attn.sr
            self.norm = attn.norm

        self.gt_num = gt_num
        self.window_size = window_size

    def forward(self, x, H, W, gt):
        B_, N_, C = x.shape
        gt_num = self.gt_num
        skip = None

        x = x.view(B_, H, W, C)
        pad_l = pad_t = 0
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C
        # x_windows = x
        B, N_, C = x_windows.shape


        if self.gt_num != 0:
            # if len(gt.shape) != 3:
            #     gt = repeat(gt, "g c -> b g c", b=B)# shape of (num_windows*B, G, C)
            nHg, nWg = gt.shape[1], gt.shape[2]
            nHp, nWp = Hp//self.window_size[0], Wp//self.window_size[1]
            if nHg != nHp or nWg != nWp:
                gt = rearrange(nn.functional.interpolate(rearrange(gt, 'b h w g c -> b g c h w'), size=(nHp, nWp) ), 'b g c h w -> b h w g c')
            gt = rearrange(gt, 'b h w g c -> (b h w) g c')
            skip = gt

            x_windows = torch.cat([gt, x_windows], dim=1)
      
        B, N, C = x_windows.shape

        q = self.q(x_windows).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            if gt_num != 0:
                x_ = x_windows[:,gt_num:,:].permute(0, 2, 1).reshape(B, C, self.window_size[0], self.window_size[1])
                x_ = nn.functional.interpolate(x_, scale_factor=self.sr_ratio, mode="bilinear", align_corners=True)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                x_ = torch.cat([gt, x_], dim=1)
            else:
                x_ = x_windows.permute(0, 2, 1).reshape(B, C, self.window_size[0], self.window_size[1])
                x_ = nn.functional.interpolate(x_, scale_factor=self.sr_ratio, mode="bilinear", align_corners=True)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # kv = self.kv(x_windows).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)


        gt = x[:,:gt_num,:]
        x = x[:,gt_num:,:]

        x = x.view(-1, self.window_size[0], self.window_size[1], C)
        x = window_reverse(x, self.window_size, Hp, Wp)  # B H' W' C

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B_, H * W, C)

        return x, gt, skip

class Block(nn.Module):

    def __init__(self, block, dim, num_heads, gt_num=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, window_size=(8,8), do_gmsa=True):
        super().__init__()
        self.norm1 = block.norm1
        self.attn = Attention(block.attn, gt_num)
        self.drop_path = block.drop_path
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.gt_num = gt_num
        self.do_gmsa = do_gmsa

        if do_gmsa:
            mlp_hidden_dim = int(mlp_ratio*dim)
            self.gt_mlp1 = ClassiqueMlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                                        act_layer=act_layer, drop=drop)

            self.gt_attn = ClassicAttention(dim=dim, num_heads=num_heads, 
                                            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            self.gt_mlp2 = ClassiqueMlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                                        act_layer=act_layer, drop=drop)
            self.gt_norm1 = norm_layer(dim)
            self.gt_norm2 = norm_layer(dim)

    def forward(self, x, H, W, gt, pe):
        
        skip = x
        x = self.norm1(x)
        gt = self.norm1(gt)
        # skip_gt = None

        B_, N_, C = x.shape
        gt_num = self.gt_num
        x = x.view(B_, H, W, C)
        pad_l = pad_t = 0
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)

        if (len(gt.shape) > 3):
            if (nHg != nHp or nWg != nWp):
                ngt=gt.shape[3]
                gt = rearrange(gt, 'b h w g c -> (b g) c h w')
                gt = nn.functional.interpolate(gt, size=(nHp, nWp))
                gt = rearrange(gt, '(b g) c h w -> b h w g c', g=ngt)
            gt = rearrange(gt, 'b h w g c -> (b h w) g c')

        nHg, nWg = pe.shape[0], pe.shape[1]
        nHp, nWp = Hp//self.window_size, Wp//self.window_size
        if (nHg != nHp or nWg != nWp):
            pe = rearrange(pe, 'h w g c -> g c h w')
            pe = nn.functional.interpolate(pe, size=(nHp, nWp))
            pe = rearrange(pe, 'g c h w -> h w g c')
        skip_gt = gt

        

        x, gt, skip_gt = self.attn(x, H, W, gt)
        B = gt.shape[0]
        # x =self.attn(x, H, W)
        x = skip + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        if self.gt_num != 0 and self.do_gmsa:
            # if len(skip_gt.shape) != 3:
            #     skip_gt = repeat(skip_gt, "g c -> b g c", b=B)
            
            gt = skip_gt + self.drop_path(gt)
            gt = gt + self.drop_path(self.gt_mlp1(self.norm2(gt)))


            # do g msa
            B, ngt, c = gt.shape
            nw = B//x.shape[0]
            gt =rearrange(gt, "(b n) g c -> b (n g) c", n=nw)

            gt = gt + self.drop_path(self.gt_attn(self.gt_norm1(gt), pe))
            gt = gt + self.drop_path(self.gt_mlp2(self.gt_norm2(gt)))
            gt = rearrange(gt, "b (n g) c -> (b n) g c",g=ngt, c=c)

        return x, gt






@BACKBONES.register_module()
class SegFormerGTZeta(nn.Module):
    """docstring for SegFormerGTZeta"""
    def __init__(self, gt_num = 10):
        super(SegFormerGTZeta, self).__init__()
        self.gt_num = gt_num
        self.embed_dims=[64, 128, 320, 512]
        self.num_heads=[1, 2, 5, 8]
        self.window_size=(8,8)


        ngt = 16 # 512//(4*8)

        ws_pe = (40*gt_num//(2**0), 40*gt_num//(2**0))
        self.global_token1 = torch.nn.Parameter(torch.randn(ngt//(2**0),ngt//(2**0),gt_num,self.embed_dims[0]))
        self.pe1 = nn.Parameter(torch.zeros(ws_pe[0]*ws_pe[1], self.embed_dims[0]))
        trunc_normal_(self.pe1, std=.02)

        ws_pe = (40*gt_num//(2**1), 40*gt_num//(2**1))
        self.global_token2 = torch.nn.Parameter(torch.randn(ngt//(2**1),ngt//(2**1),gt_num,self.embed_dims[1]))
        self.pe2 = nn.Parameter(torch.zeros(ws_pe[0]*ws_pe[1], self.embed_dims[1]))
        trunc_normal_(self.pe2, std=.02)

        ws_pe = (40*gt_num//(2**2), 40*gt_num//(2**2))
        self.global_token3 = torch.nn.Parameter(torch.randn(ngt//(2**2),ngt//(2**2),gt_num,self.embed_dims[2]))
        self.pe3 = nn.Parameter(torch.zeros(ws_pe[0]*ws_pe[1], self.embed_dims[2]))
        trunc_normal_(self.pe3, std=.02)

        ws_pe = (40*gt_num//(2**3), 40*gt_num//(2**3))
        self.global_token4 = torch.nn.Parameter(torch.randn(ngt//(2**3),ngt//(2**3),gt_num,self.embed_dims[3]))
        self.pe4 = nn.Parameter(torch.zeros(ws_pe[0]*ws_pe[1], self.embed_dims[3]))
        trunc_normal_(self.pe4, std=.02)



    def init_weights(self, pretrained=None):
        mix = mit_b4(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

        if isinstance(pretrained, str):
            mix.init_weights(pretrained)

        depths=[3, 8, 27, 3]

        self.patch_embed1 = mix.patch_embed1
        self.patch_embed2 = mix.patch_embed2
        self.patch_embed3 = mix.patch_embed3 
        self.patch_embed4 = mix.patch_embed4

        # transformer encoder
        do_gmsa = [True]*depths[0]
        do_gmsa[-1] = False
        self.block1 = nn.ModuleList([Block(mix.block1[i], gt_num=self.gt_num, dim=self.embed_dims[0],
                                           num_heads=self.num_heads[0], mlp_ratio=4., 
                                           qkv_bias=True, drop=0., attn_drop=0.,
                                           window_size=(8,8), do_gmsa=do_gmsa[i])
            for i in range(depths[0])])
        self.norm1 = mix.norm1

        do_gmsa = [True]*depths[1]
        do_gmsa[-1] = False
        self.block2 = nn.ModuleList([Block(mix.block2[i], gt_num=self.gt_num, dim=self.embed_dims[1],
                                           num_heads=self.num_heads[1], mlp_ratio=4., 
                                           qkv_bias=True, drop=0., attn_drop=0.,
                                           window_size=(8,8), do_gmsa=do_gmsa[i])
            for i in range(depths[1])])
        self.norm2 = mix.norm2

        do_gmsa = [True]*depths[2]
        do_gmsa[-1] = False
        self.block3 = nn.ModuleList([Block(mix.block3[i], gt_num=self.gt_num, dim=self.embed_dims[2],
                                           num_heads=self.num_heads[2], mlp_ratio=4., 
                                           qkv_bias=True, drop=0., attn_drop=0.,
                                           window_size=(8,8), do_gmsa=do_gmsa[i])
            for i in range(depths[2])])
        self.norm3 = mix.norm3

        do_gmsa = [True]*depths[3]
        do_gmsa[-1] = False
        self.block4 = nn.ModuleList([Block(mix.block4[i], gt_num=self.gt_num, dim=self.embed_dims[3],
                                           num_heads=self.num_heads[3], mlp_ratio=4., 
                                           qkv_bias=True, drop=0., attn_drop=0.,
                                           window_size=(8,8), do_gmsa=do_gmsa[i])
            for i in range(depths[3])])
        self.norm4 = mix.norm4

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        gt = self.global_token1
        gt = repeat(gt, 'h w g c -> b h w g c', b=B)
        for i, blk in enumerate(self.block1):
            x, gt = blk(x, H, W, gt, self.pe1)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        gt = self.global_token2
        gt = repeat(gt, 'h w g c -> b h w g c', b=B)
        for i, blk in enumerate(self.block2):
            x, gt = blk(x, H, W, gt, self.pe2)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        gt = self.global_token3
        gt = repeat(gt, 'h w g c -> b h w g c', b=B)
        for i, blk in enumerate(self.block3):
            x, gt = blk(x, H, W, gt, self.pe3)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        gt = self.global_token4
        gt = repeat(gt, 'h w g c -> b h w g c', b=B)
        for i, blk in enumerate(self.block4):
            x, gt = blk(x, H, W, gt, self.pe4)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x