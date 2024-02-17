import torch
import torch.functional as F
from einops import rearrange
from timm.models.layers import drop_path
from torch import nn


class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) with GELU activation and optional dropout.

    Parameters:
    -----------
    in_features: int
        Number of input features.
    hidden_features: int (default=None)
        Number of hidden features, if None, set to in_features.
    out_features: int (default=None)
        Number of output features, if None, set to in_features.
    act_layer: nn.GELU (default)
        Activation function.
    drop: float (default=0.0)
        Dropout rate.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TemporalConv(nn.Module):
    """
    Temporal Convolutional Module inspired by Visual Transformer.
    """

    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7)
        )
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        x = rearrange(x, "B N A T -> B (N A) T")
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, "B C NA T -> B NA (T C)")
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.
    Copied from https://github.com/facebookresearch/vissl

    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    # TO-DO: Do we need this?
    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=None,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=None,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1
            ) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads)
            )  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
            )  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(
                1, 2, 0
            ).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(
                size=(window_size[0] * window_size[1] + 1,) * 2,
                dtype=relative_coords.dtype,
            )
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1,
                -1,
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv

        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=None,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        window_size=None,
        attn_head_dim=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(
                self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True
            )
        if return_qkv:
            y, qkv = self.attn(
                self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv
            )
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv

        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(
                self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias)
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """EEG to Patch Embedding"""

    def __init__(self, EEG_size=2000, patch_size=200, in_chans=1, embed_dim=200):
        super().__init__()
        # EEG_size = to_2tuple(EEG_size)
        # patch_size = to_2tuple(patch_size)
        num_patches = 62 * (EEG_size // patch_size)
        self.patch_shape = (1, EEG_size // patch_size)
        self.EEG_size = EEG_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size)
        )

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1
        ) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1,
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
