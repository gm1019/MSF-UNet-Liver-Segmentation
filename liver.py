# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 18:46:03 2024

@author: Ming Gong
"""



import argparse
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint

from dataset.data_loader import trainGenerator
from msf_unet import build_msf_unet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MSF-UNet for liver tumour segmentation."
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Directory containing the training image and mask folders.",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        default="images",
        help="Name of the training image folder.",
    )
    parser.add_argument(
        "--mask-folder",
        type=str,
        default="masks",
        help="Name of the training mask folder.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/msf_unet_best.keras",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_dir = os.path.dirname(args.checkpoint)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    augmentation = {
        "rotation_range": 0.2,
        "width_shift_range": 0.05,
        "height_shift_range": 0.05,
        "shear_range": 0.05,
        "zoom_range": 0.05,
        "horizontal_flip": True,
        "fill_mode": "nearest",
    }

    train_gen = trainGenerator(
        batch_size=args.batch_size,
        train_path=args.train_dir,
        image_folder=args.image_folder,
        mask_folder=args.mask_folder,
        aug_dict=augmentation,
        save_to_dir=None,
    )

    model = build_msf_unet()

    callbacks = [
        ModelCheckpoint(
            args.checkpoint,
            monitor="loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="loss",
            patience=10,
            restore_best_weights=True,
        ),
    ]

    model.fit(
        train_gen,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()





