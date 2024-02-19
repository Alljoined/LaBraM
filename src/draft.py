import torch
from torch import nn
from models.labram_finetune import NeuralTransformer

# Why this input shape?
X = torch.randn(1, 64, 1, 200)
# Like, 1 batch, 128 something, 1 channel, 200 time steps ?

model = NeuralTransformer(
    EEG_size=1600,
    patch_size=200,
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
    use_rel_pos_bias=False,
    use_shared_rel_pos_bias=False,
    use_mean_pooling=True,
    init_scale=0.001,
)

model.eval()

with torch.no_grad():
    pred = model(X)


model = NeuralTransformer(
    n_times=1600,
    patch_size=200,
    n_chans=1,
    out_chans=8,
    num_classes=1000,
    emb_size=200,
    n_layers=12,
    att_num_heads=10,
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
    use_rel_pos_bias=False,
    use_shared_rel_pos_bias=False,
    use_mean_pooling=True,
    init_scale=0.001,
)
