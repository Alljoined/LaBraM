import torch
from einops import rearrange
from einops.layers.torch import Rearrange
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
        in_features: int,
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

    In this module we apply the follow steps three times repeatedly
    to the input tensor, reducing the temporal dimension only in the first.
    - Apply a 2D convolution.
    - Apply a GELU activation function.
    - Apply a GroupNorm with 4 groups.

    Parameters:
    -----------
    in_chans: int (default=1)
        Number of input channels.
    out_chans: int (default=8)
        Number of output channels.
    num_groups: int (default=4)
        Number of groups for GroupNorm.
    kernel_size_1: tuple (default=(1, 15))
        Kernel size for the first convolution.
    kernel_size_2: tuple (default=(1, 3))
        Kernel size for the second and third convolutions.
    stride_1: tuple (default=(1, 8))
        Stride for the first convolution.
    padding_1: tuple (default=(0, 7))
        Padding for the first convolution.
    padding_2: tuple (default=(0, 1))
        Padding for the second and third convolutions.
    Returns:
    --------
    x: torch.Tensor
        Output tensor of shape (Batch, NA, Temporal Channel).
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=8,
        num_groups=4,
        kernel_size_1=(1, 15),
        stride_1=(1, 8),
        padding_1=(0, 7),
        kernel_size_2=(1, 3),
        padding_2=(0, 1),
        act_layer=nn.GELU,
    ):
        super().__init__()

        # Here, we use the Rearrange layer from einops to flatten the input
        # tensor to a 2D tensor, so we can apply 2D convolutions.
        self.channel_patch_flatten = Rearrange(
            "Batch chs npat spatch -> Batch () (chs npat) spatch"
        )

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_1,
            stride=stride_1,
            padding=padding_1,
        )
        self.act_layer_1 = act_layer()
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_2,
            padding=padding_2,
        )
        self.act_layer_2 = act_layer()
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_2,
            padding=padding_2,
        )
        self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act_layer_3 = act_layer()

        self.transpose_temporal_channel = Rearrange("Batch C NA T -> Batch NA (T C)")

    def forward(self, x):
        """
        Apply 3 steps of 2D convolution, GELU activation function,
        and GroupNorm.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (Batch, Channels, n_patchs, size_patch).

        Returns:
        --------
        x: torch.Tensor
            Output tensor of shape (Batch, NA, Temporal Channel).
        """
        x = self.channel_patch_flatten(x)
        x = self.act_layer_1(self.norm1(self.conv1(x)))
        x = self.act_layer_2(self.norm2(self.conv2(x)))
        x = self.act_layer_3(self.norm3(self.conv3(x)))
        x = self.transpose_temporal_channel(x)
        return x


class PatchEmbed(nn.Module):
    """EEG to Patch Embedding"""

    def __init__(self, n_times=2000, patch_size=200, in_channels=1, embed_dim=200):
        super().__init__()
        num_patches = 62 * (n_times // patch_size)
        self.patch_shape = (1, n_times // patch_size)
        self.EEG_size = n_times
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(1, patch_size),
            stride=(1, patch_size),
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
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


# Bru's comments: I think we should split into two different classes
class Attention(nn.Module):
    """
    Attention with the options of Window-based multi-head self attention (W-MSA).

    This code is strong inspired by:
    https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L77

    Basically, the attention module is a linear layer that takes the input tensor
    and returns the output tensor. The input tensor is first passed through a linear
    layer to get the query, key, and value tensors. Then, the query tensor is multiplied
    by the scale factor and the result is multiplied by the transpose of the key tensor.

    The flag window_size is used to determine if the attention is window-based or not.

    Parameters:
    -----------
    dim: int
        Number of input features.
    num_heads: int (default=8)
        Number of attention heads.
    qkv_bias: bool (default=False)
        If True, add a learnable bias to the query, key, and value tensors.
    qk_norm: nn.LayerNorm (default=None)
        If not None, apply LayerNorm to the query and key tensors.
    qk_scale: float (default=None)
        If not None, use this value as the scale factor. If None,
        use head_dim**-0.5, where head_dim = dim // num_heads.
    attn_drop: float (default=0.0)
        Dropout rate for the attention weights.
    proj_drop: float (default=0.0)
        Dropout rate for the output tensor.
    window_size: bool (default=None)
        If not None, use window-based multi-head self attention based on Swin Transformer.
    attn_head_dim: int (default=None)
        If not None, use this value as the head_dim. If None, use dim // num_heads.
    """

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

    def forward(
        self,
        x: torch.Tensor,
        rel_pos_bias=None,
        return_attention=False,
        return_qkv=False,
    ):
        """
        Apply the attention mechanism to the input tensor.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (Batch, N, C).
        rel_pos_bias: torch.Tensor (default=None)
            If not None, add this tensor to the attention weights.
        return_attention: bool (default=False)
            If True, return the attention weights.
        return_qkv: bool (default=False)
            If True, return the query, key, and value tensors together with
            the output tensor.
        Returns:
        --------
        x: torch.Tensor
            Output tensor of shape (Batch, N, C).
        qkv: torch.Tensor (optional)
            Query, key, and value tensors of shape
            (Batch, N, 3, num_heads, C // num_heads).
        """
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
        qkv = nn.functional.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
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


# The authors first copied from timm and then modified the code.
class WindowsAttentionBlock(nn.Module):
    """Attention Block from Vision Transformer with support for
    Window-based Attention.

    Parameters:
    -----------
    dim: int
        Number of input features.
    num_heads: int (default=8)
        Number of attention heads.
    mlp_ratio: float (default=4.0)
        Ratio to increase the hidden features from input features in the MLP layer
    qkv_bias: bool (default=False)
        If True, add a learnable bias to the query, key, and value tensors.
    qk_norm: nn.LayerNorm (default=None)
        If not None, apply LayerNorm to the query and key tensors.
    qk_scale: float (default=None)
        If not None, use this value as the scale factor. If None,
        use head_dim**-0.5, where head_dim = dim // num_heads.
    drop: float (default=0.0)
        Dropout rate for the output tensor.
    attn_drop: float (default=0.0)
        Dropout rate for the attention weights.
    drop_path: float (default=0.0)
        Dropout rate for the output tensor.
    init_values: float (default=None)
        If not None, use this value to initialize the gamma_1 and gamma_2
        parameters.
    act_layer: nn.GELU (default)
        Activation function.
    norm_layer: nn.LayerNorm (default)
        Normalization layer.
    window_size: bool (default=None)
        If not None, use window-based multi-head self attention based on
        Swin Transformer.
    attn_head_dim: int (default=None)
        If not None, use this value as the head_dim. If None,
        the classes use dim // num_heads

    Returns:
    --------
    x: torch.Tensor
        Output tensor of shape (Batch, N, C). [I think]

    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
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


class SegmentPatch(nn.Module):
    """Segment and Patch for EEG data.

    Adapted Patch Embedding inspired in the Visual Transform approach
    to extract the learned segmentor, we expect get the input shape as:
    (Batch, Number of Channels, number of times points).

    We apply a 2D convolution with kernel size of (1, patch_size)
    and a stride of (1, patch_size).

    The results output shape will be:
    (Batch, Number of Channels, Number of patches, patch size).

    This way, we learned a convolution to segment the input shape.

    The number of patches is calculated as the number of samples divided
    by the patch size.

    Parameters:
    -----------
    n_times: int (default=2000)
        Number of temporal components of the input tensor.
    in_chans: int (default=1)
        number of electrods from the EEG signal
    embed_dim: int (default=200)
        Number of n_output to be used in the convolution, here,
        we used the same as patch_size.
    patch_size: int (default=200)
        Size of the patch, default is 1-seconds with 200Hz.
    Returns:
    --------
    x_patched: torch.Tensor
        Output tensor of shape (batch, n_chans, num_patches, embed_dim).
    """

    def __init__(self, n_times=2000, patch_size=200, n_chans=1, embed_dim=200):
        super().__init__()

        self.n_times = n_times
        self.patch_size = patch_size
        self.n_patchs = n_times // patch_size
        self.embed_dim = embed_dim
        self.n_chans = n_chans

        self.patcher = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=(1, self.patch_size),
            stride=(1, self.patch_size),
        )

    def forward(self, x):
        """
        Using an 1D convolution to generate segments of EEG signal.

        Parameters:
        -----------
        X: Tensor
            [batch, n_chans, n_times]

        Returns:
        --------
        X_patch: Tensor
            [batch, n_chans, n_times//patch_size, patch_size]
        """
        batch_size, _, _ = x.shape
        # Input shape: [batch, n_chs, n_times]

        # First, rearrange input to treat the channel dimension 'n_chs' as
        # separate 'dimension' in batch for Conv2d
        # This requires reshaping x to have a height of 1 for each EEG sample.

        x = rearrange(x, "batch nchans temporal -> (batch nchans) 1 1 temporal")

        # Apply the convolution along the temporal dimension
        # Conv2d output shape: [(batch*n_chs), embed_dim, 1, n_patches]
        x = self.patcher(x)

        # Now, rearrange output to get back to a batch-first format,
        # combining embedded patches with channel information
        # Assuming you want [batch, n_chs, n_patches, embed_dim]
        # as output, which keeps channel information
        # This treats each patch embedding as a feature alongside channels
        x = rearrange(
            x,
            pattern="(batch nchans) embed 1 npatchs ->  " "batch nchans npatchs embed",
            batch=batch_size,
            nchans=self.n_chans,
        )

        return x


class RelativePositionBias(nn.Module):
    """
    Relative Position Bias for Window-based multi-head self attention.

    This code is strong inspired by:
    https://github.com/SwinTransformer/Video-Swin-Transformer

    Parameters:
    -----------
    window_size: tuple
        Tuple with the size of the window.
    num_heads: int
        Number of attention heads.

    Returns:
    --------
    relative_position_bias: torch.Tensor
    """

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

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1,
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        # final output: nH, Wh*Ww, Wh*Ww
        return relative_position_bias.permute(2, 0, 1).contiguous()
