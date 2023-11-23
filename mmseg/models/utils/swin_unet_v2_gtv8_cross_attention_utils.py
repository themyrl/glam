import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np

from .swin_unet_v2_utils_gtv8 import *


class WindowCrossAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., channel_scale=2):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.channel_scale = channel_scale

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0]*2 - 1) * (2 * window_size[1]*2 - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0]).repeat_interleave(2)
        coords_w = torch.arange(self.window_size[1]).repeat_interleave(2)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords[:,:,::4] # take slices in one axis
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj_q = nn.Linear(dim // channel_scale, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim // channel_scale)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, kv, q, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # B_ = num_windows*B
        B_, N, C = kv.shape

        qB_, qN, qC = q.shape
        
        kv = self.proj_kv(kv).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q = self.proj_q(q).reshape(qB_, qN, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        k, v = kv[0], kv[1] # make torchscript happy (cannot use tensor as tuple)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            (self.window_size[0] * 2) * (self.window_size[1] * 2), self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        # if mask is not None:
        #     nW = mask.shape[0]
        #     attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        #     attn = attn.view(-1, self.num_heads, N, N)
        #     attn = self.softmax(attn)
        # else:
        attn = self.softmax(attn)
    
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(qB_, qN, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class CrossAttentionBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, skip_connection_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, residual_patch_expand=True, channel_scale=2,
                 cross_attention_weight=1.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.skip_connection_resolution = skip_connection_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.window_size_skip_co = self.window_size * 2
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.channel_scale = channel_scale
        self.cross_attention_weight = cross_attention_weight
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.proj_shortcut = nn.Linear(dim, dim // self.channel_scale)
        self.upsample_shortcut = nn.UpsamplingBilinear2d(scale_factor=(2,2))
        self.expand = PatchExpand(input_resolution, dim)

        self.residual_patch_expand = residual_patch_expand
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowCrossAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            channel_scale=channel_scale)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim // self.channel_scale)
        mlp_hidden_dim = int(dim // self.channel_scale * mlp_ratio)
        self.mlp = Mlp(in_features=dim // self.channel_scale, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

 

    def forward(self, x, x_downsampled, padwh):
        H, W = self.input_resolution
        B, L, C = x.shape

        H_d, W_d = self.skip_connection_resolution
        B_d, L_d, C_d = x_downsampled.shape

        assert L == H * W, "input feature has wrong size"
        assert L_d == H_d * W_d, "cross input feature has wrong size"

            
        
        # L = 128 * 256
        # print(L, H, W)
        #if W_d != 2*W:
            #import pdb; pdb.set_trace()
        # assert H_d == 2*H
        # assert W_d == 2*W 



        if self.residual_patch_expand:
            shortcut, Wh, Ww = self.expand(x, H, W, padwh)
        else:
            shortcut = self.proj_shortcut(x)
            shortcut = shortcut.view(B, H, W, C_d).permute(0,3,1,2)
            shortcut = self.upsample_shortcut(shortcut).permute(0,2,3,1)
            if padwh[0] != 0 or padwh[1] != 0:
                shortcut = shortcut[:,:(shortcut.shape[1]-padwh[1]),:(shortcut.shape[2]-padwh[0]),:]
                shortcut = shortcut.contiguous()
            shortcut = shortcut.view(B, L_d, C_d)

        
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        #x_downsampled = self.norm1(x_downsampled)
        x_downsampled = x_downsampled.view(B_d, H_d, W_d, C_d)
        

        # Pad x
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        _, Hp, Wp, _ = x.shape

        # Pad x_downsampled
        d_pad_l = d_pad_t = 0
        d_pad_r = (self.window_size_skip_co - W_d % self.window_size_skip_co) % self.window_size_skip_co
        d_pad_b = (self.window_size_skip_co - H_d % self.window_size_skip_co) % self.window_size_skip_co
        x_downsampled = F.pad(x_downsampled, (0, 0, d_pad_l, d_pad_r, d_pad_t, d_pad_b))

        _, Hp_d, Wp_d, _ = x_downsampled.shape
        

        # # cyclic shift
        # if self.shift_size > 0:
        #     shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        #     attn_mask = mask_matrix
        # else:
        #     shifted_x = x
        #     attn_mask = None

        # partition windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # partition windows
        x_downsampled_windows = window_partition(x_downsampled, self.window_size_skip_co)  # nW*B, window_size, window_size, C
        x_downsampled_windows = x_downsampled_windows.view(-1, (self.window_size_skip_co) * (self.window_size_skip_co), C_d)  # nW*B, window_size*window_size, C


        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, x_downsampled_windows, mask=None)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size_skip_co, self.window_size_skip_co, C_d)
        x = window_reverse(attn_windows, self.window_size_skip_co, Hp_d, Wp_d)  # B H' W' C
            
        if d_pad_r > 0 or d_pad_b > 0:
            x = x[:, :H_d, :W_d, :].contiguous()
        x = x.view(B_d, H_d * W_d, C_d)


        # FFN
        x = self.drop_path(x)*self.cross_attention_weight + shortcut
        x = x + self.drop_path(self.mlp(self.norm2(x)))*self.cross_attention_weight

        return x, H_d, W_d

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops



class BasicLayer_up_Xattn(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, skip_connection_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,
                 use_cross_attention=False, residual_patch_expand=True, channel_scale=2,
                 cross_attention_weight=1.0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.skip_connection_resolution = skip_connection_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.use_cross_attention = use_cross_attention
        self.residual_patch_expand = residual_patch_expand
        self.channel_scale = channel_scale
        self.cross_attention_weight = cross_attention_weight

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                layer = CrossAttentionBlock(dim=dim, input_resolution=input_resolution,
                                            skip_connection_resolution=skip_connection_resolution,
                                            num_heads=num_heads, window_size=window_size,
                                            shift_size=0 if (i % 2 == 0) else window_size // 2,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop, attn_drop=attn_drop,
                                            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                            norm_layer=norm_layer,
                                            residual_patch_expand=residual_patch_expand,
                                            channel_scale=self.channel_scale,
                                            cross_attention_weight=cross_attention_weight)
            else:
                layer = SwinTransformerBlock(dim=dim // 2, input_resolution=[x * 2 for x in input_resolution],
                                             num_heads=num_heads, window_size=window_size,
                                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop=drop, attn_drop=attn_drop,
                                             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                             norm_layer=norm_layer)
            self.blocks.append(layer)

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, x_downsampled, H, W, H_d, W_d, padwh):
        Hp = int(np.ceil(H_d / self.window_size)) * self.window_size
        Wp = int(np.ceil(W_d / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for inx, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.skip_connection_resolution = (H_d, W_d)
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                if inx == 0:
                    x, H, W = blk(x, x_downsampled, padwh)
                else:
                    x = blk(x, attn_mask)

        # if self.upsample is not None:
        #     x = self.upsample(x)
        # if self.upsample is not None:
        #     x_down, Wh, Ww = self.upsample(x, H, W, padwh)
        #     # Wh, Ww = (H) * 2, (W) * 2
        #     return x_down, Wh, Ww
        # else:
        #     return x, H, W
        # return x

        return x, H, W





class SwinTransformerGTV8CrossAttentionUpsampleSys(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", use_cross_attention_by_layer=[True, True, True, True],
                 residual_patch_expand=True, cross_attention_weight=1.0, gt_num=1, **kwargs):

        super().__init__()

        print("SwinTransformerCrossAttentionUpsamplingSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths,
        depths_decoder,drop_path_rate,num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.use_cross_attention_by_layer = use_cross_attention_by_layer
        self.residual_patch_expand = residual_patch_expand
        self.cross_attention_weight = cross_attention_weight

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=pretrain_img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               gt_num=gt_num,
                               id_layer=i_layer)
            self.layers.append(layer)
        
        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.layers_cross_attention_up = nn.ModuleList()
        self.layers_patch_expand = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(1,self.num_layers):
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()

            layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                     input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                       patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                     depth=depths[(self.num_layers-1-i_layer)],
                                     num_heads=num_heads[(self.num_layers-1-i_layer)],
                                     window_size=window_size,
                                     mlp_ratio=self.mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop_rate, attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                     norm_layer=norm_layer,
                                     upsample=None,
                                     use_checkpoint=use_checkpoint,
                                     gt_num=gt_num,
                                     id_layer=self.num_layers-1-i_layer)

            layer_cross_attention_up = BasicLayer_up_Xattn(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer + 1)),
                                                           input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                                             patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                                           skip_connection_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                                                       patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                                           depth=1, # Only the cross attention
                                                           num_heads=num_heads[(self.num_layers-1-i_layer)],
                                                           window_size=window_size,
                                                           mlp_ratio=self.mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                                           norm_layer=norm_layer,
                                                           residual_patch_expand=self.residual_patch_expand,
                                                           cross_attention_weight=cross_attention_weight)
            patch_expand = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                         patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                       dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer + 1)),
                                       dim_scale=2)
                
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
            self.layers_cross_attention_up.append(layer_cross_attention_up)
            self.layers_patch_expand.append(patch_expand)

        self.norm = norm_layer(self.num_features)
        self.norm_up= norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(pretrain_img_size//patch_size,pretrain_img_size//patch_size),dim_scale=4,dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    #Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x.flatten(2).transpose(1, 2))
        x_downsample = []
        x_downsample_resolutions = []
        padswh = []

        for layer in self.layers:
            # x_downsample.append(x)
            x_downsample.append(x)
            x_downsample_resolutions.append((Wh, Ww))
            # x = layer(x)
            x, Wh, Ww, padwh = layer(x, Wh, Ww)
            padswh.append(padwh)

        x = self.norm(x)  # B L C
  
        return x, x_downsample, x_downsample_resolutions, Wh, Ww, padswh

    #Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample, x_downsample_resolutions, Wh, Ww, padswh):
        # exit(0)
        # Wh, Ww = x.size(2), x.size(3)
        for inx, layer_up in enumerate(self.layers_up):

            # if len(self.layers_up)-(inx+2) >= 0:
            #     padwh = padswh[-(inx+2)]
            # else: padwh = [0,0]

            padwh = padswh[-(inx+2)]
        
            
            Wh_d, Ww_d = x_downsample_resolutions[2-inx]
            skip_co = x_downsample[2-inx]
            upsampling_blk = self.layers_cross_attention_up[inx]
            upsampling_blk.input_resolution = (Wh, Ww)
            upsampling_blk.skip_connection_resolution = (Wh_d, Ww_d)
            if self.use_cross_attention_by_layer[inx]:
                x, Wh, Ww = upsampling_blk(x, skip_co, Wh, Ww, Wh_d, Ww_d, padwh)
            else:
                x, Wh, Ww = self.layers_patch_expand[inx](x, Wh, Ww, padwh)

            x = torch.cat([x, skip_co],-1)
            x = self.concat_back_dim[inx](x)
            x, Wh, Ww = layer_up(x, Wh, Ww, padwh)

        x = self.norm_up(x)  # B L C
  
        return x, Wh, Ww

    def up_x4(self, x, H, W):
        # H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        if self.final_upsample=="expand_first":
            x = self.up(x, H, W)
            x = x.view(B,4*H,4*W,-1)
            x = x.permute(0,3,1,2) #B,C,H,W
            x = self.output(x)
            
        return x

    def forward(self, x):
        # x, x_downsample = self.forward_features(x)
        x, x_downsample, x_downsample_resolutions, Wh, Ww, padswh = self.forward_features(x)
        x, Wh, Ww = self.forward_up_features(x, x_downsample, x_downsample_resolutions, Wh, Ww, padswh)
        x = self.up_x4(x, Wh, Ww)

        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops




