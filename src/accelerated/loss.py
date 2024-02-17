"""
This script gives a wrapper for all model I/O in training.
This includes simplifying the forward and backward pass
"""

from src.engine_for_pretraining import random_masking
from utils import get_input_chans

import torch
import torch.nn.functional as F
import einops as eo


def get_loss(model, tokenizer, batch):
    """
    Get loss for model given batch and tokenizer.

    :param model: The transformer model being used for masked prediction task
    :param tokenizer: The VQVAE tokenizer
    :param batch: Batch from loader, expected as dictionary with samples and ch_names
    """
    loss_fn = F.cross_entropy

    with torch.no_grad():
        samples = batch["samples"]
        ch_names = batch["ch_names"]

        input_ch = get_input_chans(ch_names)

        samples = samples.float() / 100

        samples = eo.rearrange(samples, "B N (A T) -> B N A T", T=200)
        flat_samples = eo.rearrange(samples, "B N A T -> B (N A) T")
        bool_masked_pos = random_masking(flat_samples, mask_ratio=0.5)

        # Tokenizer TODO: Is the tokenizer fine-tuned? If so, take this out of no_grad
        input_ids = tokenizer.get_codebook_indices(samples, input_ch)
        labels = input_ids[bool_masked_pos]
        labels_sym = input_ids[~bool_masked_pos]

    # Model
    outputs = model(samples, input_ch, bool_masked_pos=bool_masked_pos)
    x_rec, x_rec_sym = outputs
    loss_rec = loss_fn(x_rec, labels)
    loss_rec_sym = loss_fn(x_rec_sym, labels_sym)
    loss = loss_rec + loss_rec_sym

    return loss
