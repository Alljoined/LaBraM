# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import math
from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from timm.models import register_model

from .functions import _cfg
from .modules import Block, PatchEmbed, TemporalConv


class NeuralTransformer(nn.Module):
    def __init__(
        self,
        EEG_size=1600,
        patch_size=200,
        n_chans=64,
        in_chans=1,
        out_chans=8,
        num_classes=1000,
        embed_dim=200,
        depth=12,
        num_heads=10,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=None,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        use_abs_pos_emb=True,
        use_mean_pooling=True,
        init_scale=0.001,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )

        self.patch_embed = (
            TemporalConv(out_chans=out_chans)
            if in_chans == 1
            else PatchEmbed(
                EEG_size=EEG_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        )
        self.time_window = EEG_size // patch_size
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, n_chans*self.time_window + 1, embed_dim),
                requires_grad=True
            )
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(
            torch.zeros(1, 16, embed_dim), requires_grad=True
        )
        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=(
                        self.patch_embed.patch_shape if in_chans != 1 else None
                    ),
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.time_embed is not None:
            trunc_normal_(self.time_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        # trunc_normal_(self.mask_token, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "time_embed"}

    def forward_features(
        self,
        x,
        input_chans=None,
        return_patch_tokens=False,
        return_all_tokens=False,
    ):
        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed_used = (
            self.pos_embed[:, input_chans]
            if input_chans is not None
            else self.pos_embed
        )
        if self.pos_embed is not None:
            pos_embed = (
                pos_embed_used[:, 1:, :]
                .unsqueeze(2)
                .expand(batch_size, -1, input_time_window, -1)
                .flatten(1, 2)
            )
            pos_embed = torch.cat(
                (pos_embed_used[:, 0:1, :].expand(
                    batch_size, -1, -1), pos_embed), dim=1
            )
            x = x + pos_embed
        if self.time_embed is not None:
            nc = n if t == self.patch_size else a
            time_embed = (
                self.time_embed[:, 0:input_time_window, :]
                .unsqueeze(1)
                .expand(batch_size, nc, -1, -1)
                .flatten(1, 2)
            )
            x[:, 1:, :] += time_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(
        self,
        x,
        input_chans=None,
        return_patch_tokens=False,
        return_all_tokens=False,
    ):
        x = self.forward_features(
            x,
            input_chans=input_chans,
            return_patch_tokens=return_patch_tokens,
            return_all_tokens=return_all_tokens,
        )
        x = self.head(x)
        return x

    def forward_intermediate(self, x, layer_id=12, norm_output=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = (
                self.pos_embed[:, 1:, :]
                .unsqueeze(2)
                .expand(batch_size, -1, self.time_window, -1)
                .flatten(1, 2)
            )
            pos_embed = torch.cat(
                (self.pos_embed[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1
            )
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = (
                self.time_embed.unsqueeze(1)
                .expand(batch_size, 62, -1, -1)
                .flatten(1, 2)
            )
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        rel_pos_bias = None
        if isinstance(layer_id, list):
            output_list = []
            for layer_idx, blk in enumerate(self.blocks):
                x = blk(x)
                # use last norm for all intermediate layers
                if layer_idx in layer_id:
                    if norm_output:
                        x_norm = self.fc_norm(self.norm(x[:, 1:]))
                        output_list.append(x_norm)
                    else:
                        output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for layer_idx, blk in enumerate(self.blocks):
                if layer_idx < layer_id:
                    x = blk(x)
                elif layer_idx == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f"Not support for layer id is {layer_id} now!")

    def get_intermediate_layers(self, x, use_last_norm=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = (
                self.pos_embed[:, 1:, :]
                .unsqueeze(2)
                .expand(batch_size, -1, self.time_window, -1)
                .flatten(1, 2)
            )
            pos_embed = torch.cat(
                (self.pos_embed[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1
            )
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = (
                self.time_embed.unsqueeze(1)
                .expand(batch_size, 62, -1, -1)
                .flatten(1, 2)
            )
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        features = []
        for blk in self.blocks:
            x = blk(x)
            if use_last_norm:
                features.append(self.norm(x))
            else:
                features.append(x)

        return features

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes)
            if num_classes > 0 else nn.Identity()
        )


@register_model
def labram_base_patch200_200(pretrained=False, **kwargs):
    model = NeuralTransformer(
        patch_size=200,
        embed_dim=200,
        depth=12,
        num_heads=10,
        mlp_ratio=4,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),  # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def labram_large_patch200_200(pretrained=False, **kwargs):
    model = NeuralTransformer(
        patch_size=200,
        embed_dim=400,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        out_chans=16,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),  # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def labram_huge_patch200_200(pretrained=False, **kwargs):
    model = NeuralTransformer(
        patch_size=200,
        embed_dim=800,
        depth=48,
        num_heads=16,
        mlp_ratio=4,
        out_chans=32,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),  # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model
