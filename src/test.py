import torch
from torch import nn

from models.labram_finetune import NeuralTransformer

# As commment in the meet, the expect input is:
# Batch size, channels, time//patch_size, patch_size

in_chans = 1  # **Not working if in_chans is different of 1. Issue with temporal_embedding.**
batch_size = 1
patch_size = 200
n_time_points_patched = 16  # Max number for patch, the value is hardcode in
# the model
EEG_size = 1600

# Generating an empty vector just to get the output.
X = torch.zeros(batch_size, in_chans, n_time_points_patched, patch_size)
# Everything is default
model = NeuralTransformer(
    EEG_size=EEG_size,
    patch_size=patch_size,
    in_chans=in_chans,
    out_chans=8,
    num_classes=1000,
    embed_dim=200,
    depth=12,
    num_heads=10,
    mlp_ratio=4.,
    qkv_bias=False,
    qk_norm=None,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.,
    norm_layer=nn.LayerNorm,
    init_values=0, # default value is not working, changed from None to zero.
    use_abs_pos_emb=False,  # Not working
    use_rel_pos_bias=False, 
    use_shared_rel_pos_bias=False,
    use_mean_pooling=True,
    init_scale=0.001,
)

with torch.no_grad():
    y_pred = model(X)
