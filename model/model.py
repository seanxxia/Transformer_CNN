
from typing import Optional
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models

from timm.models.vision_transformer import Mlp, Block, VisionTransformer, _cfg, PatchEmbed
from timm.models.layers import trunc_normal_

# class CNN(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
#         super().__init__()
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
#         # print(self.backbone)
#     def forward(self, x):
#         print("conv:",self.proj(x).shape)
#         print("flat:", self.proj(x).flatten(2).shape)
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         print("flat and trans:",x.shape)
#         x = self.backbone(x)
#         print("CNN output:",x.shape)
        
#         return x

class CNN_ViT(nn.Module):
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_chans=3, 
        num_lanes=4, 
        embed_dim=768,
        depth=12,
        num_heads=12, 
        mlp_ratio=4., 
        qkv_bias=True, 
        qk_scale=None, 
        drop_rate=0., 
        attn_drop_rate=0.,
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm,
        train=False
    ):

        super().__init__()
        self.num_lanes = num_lanes
        self.embed_dim = embed_dim
        # for CNN
        self.vgg16 = models.vgg16_bn(pretrained=True).features
        modules = list(self.vgg16.children())[:-1]
        self.backbone=nn.Sequential(*modules)
        # self.backbone = models.vgg16_bn(pretrained=True).features[:-1]
        self.proj = nn.Conv2d(512, embed_dim, kernel_size=1, stride=1)

        # for ViT
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.seq_w = img_size // patch_size
        self.seq_h = img_size // patch_size
        num_patches = self.patch_embed.num_patches

        self.token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # Pediction head
        self.patch_unembed = nn.ConvTranspose2d(embed_dim, num_lanes+1, kernel_size=patch_size, stride=patch_size)

        # Initialization
        if train:
            self.load_pretrained()
    
    def load_pretrained(self):
        model = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm)
        model.default_cfg = _cfg()
        checkpoint = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
            map_location="cpu", check_hash=True
        )        
        model.load_state_dict(checkpoint['model'])

        self.pos_embed = model.pos_embed
        self.patch_embed = model.patch_embed
        self.blocks = model.blocks
        self.norm = model.norm

    def forward_hidden(self, x):
        B = x.size(0)
        # CNN
        x = self.backbone(x)
        x = self.proj(x).flatten(2).transpose(1, 2)
        tokens = self.token.expand(B, -1, -1)
        x = torch.cat((tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x[:, 1:]
        return x
    
    def forward_head(self, h):
        B = h.size(0)
        s = h.transpose(1, 2).view(B, -1, self.seq_h, self.seq_w)
        segmentation = self.patch_unembed(s)
        return segmentation
    
    def forward(self, x):
        h = self.forward_hidden(x)
        s = self.forward_head(h)
        return s


class SCNN_ViT(nn.Module):
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_chans=3, 
        num_lanes=4, 
        embed_dim=768,
        depth=12,
        num_heads=12, 
        mlp_ratio=4., 
        qkv_bias=True, 
        qk_scale=None, 
        drop_rate=0., 
        attn_drop_rate=0.,
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm,
        train=False
    ):

        super().__init__()
        self.num_lanes = num_lanes
        self.embed_dim = embed_dim
        # for CNN
        self.vgg16 = models.vgg16_bn(pretrained=True).features
        modules = list(self.vgg16.children())[:-1]
        # -11 is 28 * 28 shape
        self.backbone=nn.Sequential(*modules)
        self.proj = nn.Conv2d(512, embed_dim, kernel_size=1, stride=1)
        # for SCNN
        self.SCNN_init()

        # for ViT
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.seq_w = img_size // patch_size
        self.seq_h = img_size // patch_size
        num_patches = self.patch_embed.num_patches

        self.token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # Pediction head
        self.patch_unembed = nn.ConvTranspose2d(embed_dim, num_lanes+1, kernel_size=patch_size, stride=patch_size)

        # Initialization
        if train:
            self.load_pretrained()
    
    def SCNN_init(self, ms_ks=5):
        # input_w, input_h = input_size
        # self.fc_input_feature = 5 * int(input_w/2) * int(input_h/2)
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down', nn.Conv2d(512, 512, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('down_up', nn.Conv2d(512, 512, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('left_right',
                                        nn.Conv2d(512, 512, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        self.message_passing.add_module('right_left',
                                        nn.Conv2d(512, 512, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        # self.layer2 = nn.Sequential(
        #     nn.Dropout2d(0.1),
        #     nn.Conv2d(64, 5, 1)  # get (nB, 5, x, x)
        # )
        # self.layer3 = nn.Sequential(
        #     nn.Softmax(dim=1),  # (nB, 5, 36, 100)
        #     nn.AvgPool2d(2, 2),  # (nB, 5, 18, 50)
        # )
        # self.fc = nn.Sequential(
        #     nn.Linear(self.fc_input_feature, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 4),
        #     nn.Sigmoid()
        # )
    def message_passing_forward(self, x):
        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        """
        Argument:
        ----------
        x: input tensor
        vertical: vertical message passing or horizontal
        reverse: False for up-down or left-right, True for down-up or right-left
        """
        nB, C, H, W = x.shape
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)

    def load_pretrained(self):
        model = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm)
        model.default_cfg = _cfg()
        checkpoint = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
            map_location="cpu", check_hash=True
        )        
        model.load_state_dict(checkpoint['model'])

        self.pos_embed = model.pos_embed
        self.patch_embed = model.patch_embed
        self.blocks = model.blocks
        self.norm = model.norm

    def forward_hidden(self, x):
        B = x.size(0)
        # CNN
        x = self.backbone(x)
        # SCN
        x = self.message_passing_forward(x)
        x = self.proj(x).flatten(2).transpose(1, 2)
        tokens = self.token.expand(B, -1, -1)
        x = torch.cat((tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x[:, 1:]
        return x
    
    def forward_head(self, h):
        B = h.size(0)
        s = h.transpose(1, 2).view(B, -1, self.seq_h, self.seq_w)
        segmentation = self.patch_unembed(s)
        return segmentation
    
    def forward(self, x):
        h = self.forward_hidden(x)
        s = self.forward_head(h)
        return s