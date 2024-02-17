from typing import Dict, Any

from dataclasses import dataclass, field, asdict
import yaml


@dataclass
class ConfigClass:
    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]):
        return cls(**cfg)

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig(ConfigClass):
    batch_size: int = 64
    target_batch: int = 64  # For gradient accum
    epochs: int = 300
    save_every: int = 20
    save_dir: str = "./out/"
    pretrained_path: str = "./checkpoints/labram-base.pth"
    resume: bool = False  # Resume a run from checkpoint dir?

    # Dataloader settings
    dataloader_kwargs = {"num_workers": 1, "pin_memory": False}
    # Optimizer is AdamW as default, no need to support others currently
    opt: str = "AdamW"
    opt_kwargs: dict = {
        "lr": 5e-4,
        "weight_decay": 0.05,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
    scheduler: str = "StepLR"
    scheduler_kwargs: dict = {"step_size": 1, "gamma": 0.1}
    grad_clip: float = None
    seed: int = 0


@dataclass
class LoggingConfig(ConfigClass):
    # Use WANDB
    run_name: str = None
    wandb_entity: str = None
    wandb_project: str = None


@dataclass
class ModelConfig(ConfigClass):
    model: str = "lambram_base_patch200_1600_8k_vocab"
    rel_pos_bias: bool = False
    abs_pos_emb: bool = True
    layer_scale_init_value: float = 0.1
    input_size: int = 1600
    drop_path: float = 0.1


@dataclass
class TokenizerConfig(ConfigClass):
    tokenizer_path: str = "./checkpoints/vqnsp.pth"
    tokenizer_model: str = "vqnsp_encoder_base_decoder_3x200x12"
    codebook_size: int = 8192
    codebook_dim: int = 32


def load_yaml(yml_fp: str) -> Dict[str, ConfigClass]:
    with open(yml_fp, mode="r") as file:
        config = yaml.safe_load(file)
    d = {}
    if config["model"]:
        d["model"] = ModelConfig.from_dict(config["model"])
    if config["train"]:
        d["train"] = TrainConfig.from_dict(config["train"])
    if config["tokenizer"]:
        d["tokenizer"] = TokenizerConfig.from_dict(config["tokenizer"])
    if config["logging"]:
        d["logging"] = LoggingConfig.from_dict(config["logging"])

    return d


@dataclass
class LaBraMConfig(ConfigClass):
    train: TrainConfig
    model: ModelConfig
    logging: LoggingConfig
    tokenizer: TokenizerConfig

    @classmethod
    def load_yaml(cls, yml_fp: str):
        with open(yml_fp, mode="r") as file:
            config = yaml.safe_load(file)
        return cls.from_dict(config)
