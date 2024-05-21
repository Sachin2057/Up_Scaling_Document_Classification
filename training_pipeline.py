from generate_dataset import generate_dataset
from py_modules.trainer import training
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=False)
    args = parser.parse_args()
    generate_dataset()
    training(args.model, checkpoint=args.checkpoint)
