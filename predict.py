
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 18:46:03 2024

@author: Ming Gong
"""

import argparse
import os

from keras.models import load_model

from dataset.data_loader import (
    bce_dice_loss,
    dice_loss,
    generate_filenames_from_folder,
    saveResult,
    testGenerator,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MSF-UNet inference."
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(
        args.model,
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_loss": dice_loss,
        },
    )

    filenames = generate_filenames_from_folder(
        args.input_dir,
        extension=".png",
    )

    test_gen = testGenerator(
        args.input_dir,
        filenames,
        target_size=(512, 512),
    )

    predictions = model.predict(
        test_gen,
        steps=len(filenames),
        verbose=1,
    )

    saveResult(
        args.output_dir,
        predictions,
        filenames,
    )


if __name__ == "__main__":
    main()
