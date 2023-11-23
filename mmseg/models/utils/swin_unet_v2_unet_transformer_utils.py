import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import math

from .swin_unet_v2_utils import *


def positionalencoding2d(d_model, height, width):
    """
    Code from https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class UNetTransformerAttn(nn.Module):

    def __init__(self, dim, input_resolution, skip_connection_resolution, num_heads, stride=1, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.skip_connection_resolution = skip_connection_resolution
        self.num_heads = num_heads
        self.stride = stride
        head_dim = dim // num_heads


        # self.pos_encoding_W = input_resolution[0] // stride
        # self.pos_encoding_H = input_resolution[1] // stride
        
        # # define a parameter table of relative position bias
        # self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.pos_encoding_W*2 - 1) * (2 * self.pos_encoding_H*2 - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # # get pair-wise relative position index for each token inside the window
        # coords_h = torch.arange(self.pos_encoding_W).repeat_interleave(2)
        # coords_w = torch.arange(self.pos_encoding_H).repeat_interleave(2)
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        
        # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # relative_coords = relative_coords[:,:,::4] # take slices in one axis
        # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # relative_coords[:, :, 0] += self.pos_encoding_W - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.pos_encoding_H - 1
        # relative_coords[:, :, 0] *= 2 * self.pos_encoding_H - 1
        # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # self.register_buffer("relative_position_index", relative_position_index)

        self.proj_q = nn.Conv2d(dim, dim, kernel_size=stride, stride=stride, bias=qkv_bias)
        self.proj_k = nn.Conv2d(dim, dim, kernel_size=stride, stride=stride, bias=qkv_bias)
        
        self.proj_v = nn.Conv2d(dim // 2, dim, kernel_size=stride*2, stride=stride*2, bias=qkv_bias)
        
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)


        
    def forward(self, q, k, v, mask=None):
        qB, qC, qH, qW = q.shape
        kB, kC, kH, kW = k.shape
        vB, vC, vH, vW = v.shape
        
        q = q + positionalencoding2d(qC, qH, qW).unsqueeze(0).cuda()
        k = k + positionalencoding2d(kC, kH, kW).unsqueeze(0).cuda()
        v = v + positionalencoding2d(vC, vH, vW).unsqueeze(0).cuda()
        
        q = self.proj_q(q).view(qB, qC, -1)
        qB, qC, qN = q.shape    
        q = q.reshape(qB, self.num_heads, qC // self.num_heads, qN).permute(0, 1, 3, 2)
        
        k = self.proj_k(k).view(kB, kC, -1)
        kB, kC, kN = k.shape
        k = k.reshape(kB, self.num_heads, kC // self.num_heads, kN).permute(0, 1, 3, 2)
        
        v = self.proj_v(v).view(vB, kC, -1)
        vB, vC, vN = v.shape
        v = v.reshape(vB, self.num_heads, kC // self.num_heads, vN).permute(0, 1, 3, 2)
            

        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     (self.pos_encoding_W * 2) * (self.pos_encoding_H * 2),
        #     self.pos_encoding_W * self.pos_encoding_H,
        #     -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        #attn = attn + relative_position_bias.unsqueeze(0)


        x = (attn @ v).transpose(1, 2).reshape(qB, qN, qC)
        x = self.proj(x)

        x = x.permute(0, 2, 1).view(qB, qC, self.input_resolution[0] // self.stride, self.input_resolution[1] // self.stride).contiguous()

        return x



class UNetTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, skip_connection_resolution, num_heads, stride=1, qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.input_resolution = input_resolution
        self.skip_connection_resolution = skip_connection_resolution

        self.attn = UNetTransformerAttn(dim, input_resolution, skip_connection_resolution, num_heads, stride=stride, qkv_bias=qkv_bias)

        self.conv_out = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.norm = nn.BatchNorm2d(dim // 2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=(2*stride,2*stride))
        self.sig = nn.Sigmoid()
 

    def forward(self, x, skip, padwh):
        H, W = self.input_resolution
        B, L, C = x.shape
        
        H_d, W_d = self.skip_connection_resolution
        B_d, L_d, C_d = skip.shape

        x = x.view(B, H, W, C).permute(0,3,1,2).contiguous()
        skip = skip.view(B_d, H_d, W_d, C_d).permute(0,3,1,2).contiguous()

        # Padding skip co
        p_skip = F.pad(skip, (0, padwh[0], 0, padwh[1]))

        self.attn.input_resolution = (H,W)
        x = self.attn(q=x,
                      k=x,
                      v=p_skip)

        x = self.conv_out(x)
        x = self.norm(x)
        x = self.sig(x)
        x = self.up(x)

        # Unpad result
        if padwh[0] != 0 or padwh[1] != 0:
            x = x[:,:,:(x.shape[2]-padwh[1]),:(x.shape[3]-padwh[0])]
        if x.shape[2] < skip.shape[2] or x.shape[3] < skip.shape[3]:
            x = F.pad(x, (0, skip.shape[3]-x.shape[3], 0, skip.shape[2]-x.shape[2]))

        skip = (x * skip).permute(0, 2, 3, 1).view(B_d, -1, C_d)

        return skip

    



class SwinTransformerUNetTransformerUpsampleSys(nn.Module):
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
                 use_checkpoint=False, final_upsample="expand_first", use_cross_attention_by_layer=[True, True, True, True], **kwargs):

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
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        
        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.layers_unet_transformer = nn.ModuleList()
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
                                     use_checkpoint=use_checkpoint)

            unet_transformer_block = UNetTransformerBlock(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer + 1)),
                                                          input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                                            patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                                          skip_connection_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer - 1 )),
                                                                                      patches_resolution[1] // (2 ** (self.num_layers-1-i_layer - 1 ))),
                                                          num_heads=num_heads[(self.num_layers-1-i_layer)],
                                                          stride=2*(i_layer-1) if 2*(i_layer-1) != 0 else 1,
                                                          qkv_bias=qkv_bias)
            print('STRIDE : ', 2*(i_layer-1) if 2*(i_layer-1) != 0 else 1)
            
            patch_expand = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                         patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                       dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer + 1)),
                                       dim_scale=2)
                
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
            self.layers_unet_transformer.append(unet_transformer_block)
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
            if self.use_cross_attention_by_layer[inx]:
                cross_attention_block = self.layers_unet_transformer[inx]
                cross_attention_block.input_resolution = (Wh, Ww)
                cross_attention_block.skip_connection_resolution = (Wh_d, Ww_d)
                skip_co = cross_attention_block(x, skip_co, padwh)

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




