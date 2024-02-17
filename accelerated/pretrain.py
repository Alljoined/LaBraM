from .configs import LaBraMConfig
from .utils import cond_iew
from .loss import get_loss

from ..modeling_pretrain import NeuralTransformerForMEM
from ..modeling_vqnsp import *
from ..utils import build_pretraining_dataset

from accelerate import Accelerator, init_empty_weights
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm

class Trainer:
    def __init__(self, config : LaBraMConfig):
        self.config = config

        # Grad accum
        if self.config.train.target_batch is not None:
            self.accum_steps = self.config.train.target_batch // self.config.train.batch_size
        else:
            self.accum_steps = 1

        self.accelerator = Accelerator(
            log_with = "wandb",
            gradient_accumulation_steps = self.accum_steps
        )

        # Logging
        tracker_kwargs = {}
        self.use_wandb = not (config.logging.wandb_project is None)
        if self.use_wandb:
            log = config.logging
            tracker_kwargs["wandb"] = {
                "name" : log.run_name,
                "entity" : log.wandb_entity,
                "mode" : "online"
            }

            self.accelerator.init_trackers(
                project_name = log.wandb_project,
                config = config.to_dict(),
                init_kwargs = tracker_kwargs
            )

        self.world_size = self.accelerator.state.num_processes

        self.model = setup_model()
        self.tokenizer = setup_tokenizer()

        if self.config.train.resume:
            self.accelerator.load_state(self.config.train.save_dir)

    def setup_model(self):
        """
        Setup model using self.config
        """
        model_config = self.config.model
        name = self.config.model.model
        tokenizer_config = self.config.tokenizer

        vocab_size = tokenizer_config.codebook_size
        if vocab_size is None:
            vocab_size = 8192
        
        path = self.config.train.pretrained_path
        
        with cond_iew(path is not None):
            if name == "labram_base_patch200_1600_8k_vocab":
                model = NeuralTransformerForMEM(
                    patch_size=200, embed_dim=200, depth=12, num_heads=12, mlp_ratio=4, 
                    qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=8,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size
                )
            elif name == "labram_large_patch200_1600_8k_vocab":
                model = NeuralTransformerForMEM(
                    patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, 
                    qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=16,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size
                )
            elif name == "labram_huge_patch200_1600_8k_vocab":
                model = NeuralTransformerForMEM(
                    patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, 
                    qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=32,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size
                )

        if path is not None:
            cp = torch.load(
                path, map_location = "cuda:0" # Assume cuda:0 for training
                # Accelerate will fix the map location later anyways
            )
            model.load_state_dict(cp['model'])

        return model
    
    def setup_tokenizer(self):
        tokenizer_name = self.config.tokenizer.tokenizer_model
        pretrained_weight = self.config.tokenizer.tokenizer_path
        n_code = self.config.tokenizer.codebook_size
        code_dim = self.config.tokenizer.codebook_dim

        if tokenizer_name == "vqnsp_encoder_base_decoder_3x200x12":
            tokenizer = vqnsp_encoder_base_decoder_3x200x12(pretrained=True, pretrained_weight=pretrained_weight, as_tokenzer=True, n_code=n_code, code_dim=code_dim)
        elif tokenizer_name == "vqnsp_encoder_large_decoder_3x200x24":
            tokenizer = vqnsp_encoder_large_decoder_3x200x24(pretrained=True, pretrained_weight=pretrained_weight, as_tokenzer=True, n_code=n_code, code_dim=code_dim)

        return tokenizer

    def is_main(self): # Is main accelerate process
        return self.accelerator.is_main_process

    def train(self, loader):
        # TODO: Add their WD scheduler
        # TODO: Loader should be setup to give
        # "sample" and "ch_names" in a dictionary

        optimizer = getattr(torch.optim, self.config.train.opt)(
            self.model.parameters(), 
            **self.config.train.opt_kwargs
        )
        scheduler = getattr(torch.optim.lr_scheduler, self.config.train.scheduler)(
            optimizer, 
            **self.config.train.scheduler_kwargs
        )

        self.model, optimizer, scheduler, loader = self.accelerator.prepare(
            self.model, optimizer, scheduler, loader
        )

        # Does tokenizer need to be prepared the same?
        self.tokenizer = self.accelerator.prepare(
            self.tokenizer
        )

        for epoch in range(self.config.train.epochs):
            for idx, batch in tqdm(enumerate(loader), disable = not self.is_main()):
                with self.accelerator.accumulate(self.model), self.accelerator.autocast():
                    optimizer.zero_grad()
                    loss = get_loss(self.model, self.tokenizer, batch)
                    self.accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()

                    self.accelerator.wait_for_everyone()

                    # Logging
                    if self.use_wandb:
                        self.accelerator.log({
                            "loss" : loss.item()
                        })
                    
                    # Saving
                    if idx % self.config.train.save_every == 0:
                        self.accelerator.save_state(
                            output_dir = self.config.train.save_dir
                        )



