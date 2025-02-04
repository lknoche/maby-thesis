"""
The same testing script as previously, 
but we look at the end of the experiment rather than the beginning.
"""

import argparse
import logging
from test import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gfp_type", type=str, required=True)
    parser.add_argument("-t", "--train_date", type=str, required=True)
    parser.add_argument("-p", "--test_position", type=int, default=5)
    parser.add_argument("-s", "--start_tp", type=int, default=100)
    parser.add_argument("-b", "--test_batch_size", type=int, default=16)
    parser.add_argument("-d", "--unet_depth", type=int, default=3)
    parser.add_argument("-i", "--in_channels", type=int, default=5)
    parser.add_argument("-o", "--out_channels", type=int, default=5)
    parser.add_argument("-m", "--model_checkpoint", type=str, default="best_model.pth")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
    )
    args = parse_args()
    # We're going to look at multiple positions, so we need to change the filename
    test_filename = f"generalization_test_{args.test_position}.csv"
    evaluate(
        args.gfp_type,
        args.train_date,
        test_position=args.test_position,
        start_tp=args.start_tp,
        end_tp=None,  # Go to the end of the experiment
        test_batch_size=args.test_batch_size,
        unet_depth=args.unet_depth,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        model_checkpoint=args.model_checkpoint,
        output_file_name=test_filename,
    )
