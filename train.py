# -*- coding: utf-8 -*-
# @File : train.py
# @Author : Kaicheng Yang
# @Time : 2022/01/26 11:03:11
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import CAFIA_Transformer
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from PIL import Image
import logging
import torch.distributed as dist
from logger import setup_worker_logging
from scheduler import cosine_lr
from apex import amp
logging.basicConfig(level = logging.NOTSET)
 
def train_model(gpu, args, log_queue):
    torch.manual_seed(0)
    args.gpu = gpu
    args.rank = gpu
    setup_worker_logging(args.rank, log_queue, args.log_level)
    normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = transforms.Compose([
            Resize((224, 224)),
            ToTensor(),
            normalize,
        ])

    train_dataset = datasets.CIFAR10(root = '/home/yangkaicheng/image-classification-fenbushi/data', train = True,
                                        download = False, transform = transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=args.rank)

    trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=train_sampler)

    test_dataset = datasets.CIFAR10(root = '/home/yangkaicheng/image-classification-fenbushi/data', train = False,
                                        download = False, transform = transform)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=args.rank)
                                                                       
    testloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=test_sampler)


    dist.init_process_group(
            backend=args.dist_backend,#是通信所用的后端，可以是"ncll" "gloo"或者是一个torch.distributed.Backend类
            init_method=args.dist_url,#这个URL指定了如何初始化互相通信的进程
            world_size=args.world_size,#分布式训练所有进程数目=GPU个数
            rank=args.rank,#进程的编号，也是其优先级
        )
    torch.cuda.set_device(args.gpu)
    model = CAFIA_Transformer(args)
    model.cuda(args.gpu)
    
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), betas=(args.beta1, args.beta2), eps = args.eps, lr = args.learning_rate, weight_decay = args.weight_decay)
    
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    
    total_steps = (len(trainloader) // args.batch_size + 1)  * args.epoches
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)   
    nb_tr_steps = 0
    model.train()
    for epoch in range(args.epoches):
        train_loss = 0 
        train_iter = 0
        for _, batch in enumerate(tqdm(trainloader, desc = "Iteration")):
            nb_tr_steps += 1  
            optimizer.zero_grad()
            batch_X, batch_Y = batch
            if args.gpu is not None:
                batch_X = batch_X.cuda(non_blocking=True)
                batch_Y = batch_Y.cuda(non_blocking=True)
            outputs = model(batch_X)
            loss = criteria(outputs, batch_Y)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            scheduler(nb_tr_steps)
            train_loss += loss.item()
            train_iter += 1
            logging.info('Epoch:%d batch_loss:%f', epoch, loss)
        train_loss = train_loss / train_iter

        #eval
        model.eval()
        total, correct = 0, 0
        for _, batch in enumerate(tqdm(testloader, desc = "Iteration")):
            batch_X, batch_Y = batch
            if args.gpu is not None:
                batch_X = batch_X.cuda(args.gpu, non_blocking=True)
                batch_Y = batch_Y.cuda(args.gpu, non_blocking=True)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_Y.size(0)
            correct += (predicted == batch_Y).sum()
            
        acc = (correct / total).item()
        if args.rank == 0:
            logging.info('Epoch: %d train_loss: %f Accuracy: %f', epoch, train_loss, acc)
            output_path = os.path.join(args.output, str(acc)+'.pt')
            torch.save(model.state_dict(), output_path)




     

