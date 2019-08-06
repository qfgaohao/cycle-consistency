from vision.nets import (
    unet128_cycle_gan, unet256_cycle_gan, simple_adam_optimizer, 
    residual_unet128_cycle_gan, residual_unet256_cycle_gan
)
from vision.datasets import SimpleImageFolder
from vision.utils import Accumulator

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import utils
import torch
import os
import logging
from PIL import Image
import shutil
from torch.utils.tensorboard import SummaryWriter



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='CycleGan Training.')

    parser.add_argument('-d', '--data-path', help='dataset')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument("-s", "--sample-path", default="./samples")

    parser.add_argument("-g", "--generator", default="residual_unet256", 
        help="choose from unet128, unet256, residual_unet128 and residual_unet256.")

    parser.add_argument('-i', '--input-size', default=256, type=int, metavar='N')

    parser.add_argument('-lr', "--lr", default=0.001, type=float)

    parser.add_argument('-li', '--lambda-identity', default=5, type=int)
    parser.add_argument('-lc', '--lambda-cycle', default=10, type=int)

    return parser.parse_args()


def make_gan(name):
    if name == 'unet128':
        return unet128_cycle_gan()
    elif name == 'unet256':
        return unet256_cycle_gan()
    
    elif name == 'residual_unet128':
        return residual_unet128_cycle_gan()
    elif name == 'residual_unet256':
        return residual_unet256_cycle_gan()
    else:
        raise NameError(f"{name} is supported.")


def main(args):
    if os.path.exists(args.sample_path):
        shutil.rmtree(args.sample_path)
    os.makedirs(args.sample_path)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0)),
        transforms.ToTensor()
    ])
    path = os.path.abspath(os.path.expanduser(args.data_path))
    data_A = SimpleImageFolder(os.path.join(path, "trainA"), transform)
    data_B = SimpleImageFolder(os.path.join(path, "trainB"), transform)
    logging.info(f"Data size: ({len(data_A)}, {len(data_B)})")
    loader_A = DataLoader(data_A, batch_size=args.batch_size, shuffle=True)
    loader_B = DataLoader(data_B, batch_size=args.batch_size, shuffle=True)
    gan = make_gan(args.generator)
    gan = gan.to(DEVICE)

    optimizer = simple_adam_optimizer(gan, args.lr, 
        lambda_identity=args.lambda_identity,
        lambda_cycle=args.lambda_cycle)
    optimizer = optimizer.to(DEVICE)

    writer = SummaryWriter()
    for epoch in range(args.epochs):
        acc = Accumulator()
        for i, (input_A, input_B) in enumerate(zip(loader_A, loader_B)):
            if input_A.size(0) != input_B.size(0):
                continue

            input_A = input_A.to(DEVICE)
            input_B = input_B.to(DEVICE)

            if i == 0:
                fake_B, reconstructed_A, fake_A, reconstructed_B = gan(input_A, input_B)
                data = torch.stack([input_A[0], fake_B[0], reconstructed_A[0], 
                                   input_B[0], fake_A[0], reconstructed_B[0]], 0)
                image = utils.make_grid(data, nrow=3)
                writer.add_image('images', image, epoch)
                
                utils.save_image(data, os.path.join(args.sample_path, f"epoch-{epoch}.jpg"), nrow=3)

            loss = optimizer.optimize(input_A, input_B)
            acc.update(loss)
        logging.info(f"Epoch: {epoch}, {acc}.")
        for k, v in acc.mean().items():
            writer.add_scalar(k, v, global_step=epoch)



if __name__ == "__main__":
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    args = parse_args()
    main(args)           
