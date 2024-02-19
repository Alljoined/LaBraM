"""
Author: Bruno Aristimunha <b.aristimunha@gmail.com>
        Shahbuland Matiana

"""

import argparse

from accelerated.pretrain import Trainer as PretrainTrainer
from accelerated.configs import LaBraMConfig


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=["brain2image"])
    args = parser.parse_args()
    return args


def main(args):

    config = LaBraMConfig.load_yaml("configs/vqnsp.yml")
    trainer = PretrainTrainer(config)

    train_loader = ""  # get_train_loader(args.dataset)
    # Train the model
    trainer.train(loader=train_loader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
