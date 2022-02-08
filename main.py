# -*- coding: utf-8 -*-
# @File : main.py
# @Author : Kaicheng Yang
# @Time : 2022/01/26 11:03:50
import argparse
import logging
from torchvision import datasets, transforms
import torch
from torchvision.transforms import Resize, ToTensor, Normalize
from PIL import Image
from train import train_model
from logger import setup_primary_logging, setup_worker_logging
import torch.multiprocessing as mp


def main():
    parser = argparse.ArgumentParser()
    # Optimizer parameters
    parser.add_argument("--output", default = '/home/yangkaicheng/image-classification-fenbushi/output', type = str)
    parser.add_argument("--log_path", default = './log/out.log', type = str)
    parser.add_argument("--learning_rate", default = 2e-5, type = float,
                        help = "The initial learning rate for Adam.5e-5")
    parser.add_argument('--opt-eps', default = None, type = float, metavar = 'EPSILON',
                        help = 'Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument("--beta1", type = float, default = 0.99, help = "Adam beta 1.")
    parser.add_argument("--beta2", type = float, default = 0.99, help = "Adam beta 2.")
    parser.add_argument("--eps", type = float, default = 1e-6, help = "Adam epsilon.")
    parser.add_argument('--momentum', type = float, default = 0.9, metavar = 'M',
                        help = 'Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type = float, default = 2e-5,
                        help = 'weight decay (default: 2e-5)')
    parser.add_argument(
        "--warmup", type = int, default = 500, help = "Number of steps to warmup for."
    )
    parser.add_argument("--batch_size", type = int, default = 24, help = "Number of steps to warmup for.")
    parser.add_argument("--epoches", type = int, default = 5, help = "Number of steps to warmup for.")
    #Vit params
    parser.add_argument("--vit_model", default = '/home/yangkaicheng/image-classification-fenbushi/weights/imagenet21k+imagenet2012_ViT-B_16-224.pth', type = str)
    parser.add_argument("--image_size", type = int, default = 224, help = "input image size", choices = [224, 384])
    parser.add_argument("--num-classes", type = int, default = 10, help = "number of classes in dataset")
    parser.add_argument("--patch_size", type = int, default = 16)
    parser.add_argument("--emb_dim", type = int, default = 768)
    parser.add_argument("--mlp_dim", type = int, default = 3072)
    parser.add_argument("--num_heads", type = int, default = 12)
    parser.add_argument("--num_layers", type = int, default = 12)
    parser.add_argument("--attn_dropout_rate", type = float, default = 0.0)
    parser.add_argument("--dropout_rate", type = float, default = 0.1)
    
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:6100",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=8,
        help="Specify a single GPU to run the code on for debugging."
        "Leave at None to use all available GPUs.",
    )

    args = parser.parse_args()
    
    torch.multiprocessing.set_start_method("spawn")
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    args.log_level = logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level)
    mp.spawn(train_model, nprocs=ngpus_per_node, args=(args, log_queue))



if __name__ == '__main__':
    main()
    


    