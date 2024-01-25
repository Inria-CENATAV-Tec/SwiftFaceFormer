import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss, MSELoss, CosineEmbeddingLoss

import sys
sys.path.append('/Documents/PocketNet/')

from utils import losses
from config.config_distillation import config as cfg
from utils.dataset import MXFaceDataset, DataLoaderX
from utils.utils_callbacks import CallBackVerification, CallBackLoggingKD, CallBackModelCheckpointKD
from utils.utils_logging import AverageMeter, init_logging

from backbones.iresnet import iresnet100
from backbones.augment_cnn import AugmentCNN
from backbones import SwiftFormer_XS, SwiftFormer_L3, SwiftFormer_XXS
import backbones.genotypes as gt

torch.backends.cudnn.benchmark = True

def CosineSimilarityLoss(feat1, feat2, offset=0, scale=1):
    # minimize average cosine similarity
    return ((-1 * F.cosine_similarity(feat1, feat2).mean())+offset) * scale

def main(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if not os.path.exists(cfg.output) and rank == 0:
        os.makedirs(cfg.output)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)

    trainset = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)

    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=True)

    # load teacher model
    backbone_teacher = SwiftFormer_L3(distillation=False, num_classes=0).to(local_rank) #iresnet100(num_features=cfg.embedding_size).to(local_rank)
    try:
        backbone_teacher_pth = os.path.join(cfg.teacher_pth, "458592backbone.pth")
        print(backbone_teacher.load_state_dict(torch.load(backbone_teacher_pth, map_location=torch.device(local_rank))))

        if rank == 0:
            logging.info("backbone teacher loaded successfully!")
    except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        logging.info("load teacher backbone init, failed!")

    # load model
    if args_.network_student == "SwiftFormer_XS":
        backbone_student = SwiftFormer_XS(distillation=True, num_classes=0).to(local_rank) #models.get_model(args_.network_student)
    elif args_.network_student == "SwiftFormer_L3":
        backbone_student = SwiftFormer_L3(distillation=False, num_classes=0).to(local_rank) #models.get_model(args_.network_student)
    elif args_.network_student == "SwiftFormer_XXS":
        backbone_student = SwiftFormer_XXS(distillation=True, num_classes=0).to(local_rank) #models.get_model(args_.network_student)
    else:
        genotype = gt.from_str(cfg.genotypes["softmax_casia"])
        backbone_student = AugmentCNN(C=cfg.channel, n_layers=cfg.n_layers, genotype=genotype, stem_multiplier=4, emb=cfg.embedding_size).to(local_rank)
    if args.pretrained_student:
        try:
            backbone_student_pth = os.path.join(cfg.student_pth, str(cfg.student_global_step) + "backbone.pth")
            backbone_student.load_state_dict(torch.load(backbone_student_pth, map_location=torch.device(local_rank)))

            if rank == 0:
                logging.info("backbone student loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("load student backbone init, failed!")

    if args.resume:
        try:
            backbone_student_pth = os.path.join(cfg.output, str(cfg.global_step) + "backbone.pth")
            backbone_student.load_state_dict(torch.load(backbone_student_pth, map_location=torch.device(local_rank)))

            if rank == 0:
                logging.info("backbone student resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("load student backbone resume init, failed!")

    #for ps in backbone_teacher.parameters():
    #    dist.broadcast(ps, 0)

    for ps in backbone_student.parameters():
        dist.broadcast(ps, 0)

    backbone_teacher = DistributedDataParallel(
        module=backbone_teacher, broadcast_buffers=False, device_ids=[local_rank])
    backbone_teacher.eval()

    backbone_student = DistributedDataParallel(
        module=backbone_student, broadcast_buffers=False, device_ids=[local_rank])
    backbone_student.train()

    # get header
    if args.loss == "ArcFace":
        header = losses.ArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(local_rank)
    #elif args.loss == "CosFace":
    #    header = losses.MarginCosineProduct(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=64, m=cfg.margin).to(local_rank)
    #elif args.loss == "Softmax":
    #    header = losses.ArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=64.0, m=0).to(local_rank)
    else:
        print("Header not implemented")
   
    if args.resume:
        try:
            header_pth = os.path.join(cfg.output, str(cfg.global_step) + "header.pth")
            header.load_state_dict(torch.load(header_pth, map_location=torch.device(local_rank)))

            if rank == 0:
                logging.info("header resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("header resume init, failed!")
    
    header = DistributedDataParallel(
        module=header, broadcast_buffers=False, device_ids=[local_rank])
    header.train()

    opt_backbone_student = torch.optim.SGD(
        params=[{'params': backbone_student.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)
    opt_header = torch.optim.SGD(
        params=[{'params': header.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    scheduler_backbone_student = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone_student, lr_lambda=cfg.lr_func)
    scheduler_header = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_header, lr_lambda=cfg.lr_func)        

    criterion = CrossEntropyLoss()
    
    criterion2 = CosineSimilarityLoss

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank == 0: logging.info("Total Step is: %d" % total_step)

    if args.resume:
        rem_steps = (total_step - cfg.global_step)
        cur_epoch = cfg.num_epoch - int(cfg.num_epoch / total_step * rem_steps)
        logging.info("resume from estimated epoch {}".format(cur_epoch))
        logging.info("remaining steps {}".format(rem_steps))
        
        start_epoch = cur_epoch
        scheduler_backbone_student.last_epoch = cur_epoch
        scheduler_header.last_epoch = cur_epoch

        # --------- this could be solved more elegant ----------------
        opt_backbone_student.param_groups[0]['lr'] = scheduler_backbone_student.get_lr()[0]
        opt_header.param_groups[0]['lr'] = scheduler_header.get_lr()[0]

        print("last learning rate: {}".format(scheduler_header.get_lr()))
        # ------------------------------------------------------------

    callback_verification = CallBackVerification(cfg.eval_step, rank, cfg.val_targets, cfg.rec) # 2000
    callback_logging = CallBackLoggingKD(50, rank, total_step, cfg.batch_size, world_size, writer=None)
    callback_checkpoint = CallBackModelCheckpointKD(rank, cfg.output)

    loss = AverageMeter()
    loss1 = AverageMeter()
    loss2 = AverageMeter()
    global_step = cfg.global_step

    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        for step, (img, label) in enumerate(train_loader):
            global_step += 1
            img = img.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)

            with torch.no_grad():
                features_teacher = F.normalize(backbone_teacher(img))
            features_student_raw = backbone_student(img)
            features_student = F.normalize(features_student_raw[0]),F.normalize(features_student_raw[1]) 

            thetas = header(features_student[0], label)
            loss_v1 = 0.5 * criterion(thetas, label)
            loss_v2 = 0.5 * criterion2(features_student[1], features_teacher, offset=1, scale=64)
            loss_v = loss_v1 + loss_v2
            loss_v.backward()

            clip_grad_norm_(backbone_student.parameters(), max_norm=5, norm_type=2)

            opt_backbone_student.step()
            opt_header.step()

            opt_backbone_student.zero_grad()
            opt_header.zero_grad()

            loss.update(loss_v.item(), 1)
            loss1.update(loss_v1.item(), 1)
            loss2.update(loss_v2.item(), 1)
            
            callback_logging(global_step, loss, loss1, loss2, epoch)
            callback_verification(global_step, backbone_student)

        scheduler_backbone_student.step()
        scheduler_header.step()

        callback_checkpoint(global_step, backbone_student, header)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PoketNet Training with template knowledge distillation')
    parser.add_argument('--local-rank', type=int, default=0, help='local_rank')
    parser.add_argument('--network_student', type=str, default="SwiftFormer_XXS", help="backbone of student network")
    parser.add_argument('--network_teacher', type=str, default="SwiftFormer_L3", help="backbone of teacher network")
    parser.add_argument('--loss', type=str, default="ArcFace", help="loss function")
    parser.add_argument('--pretrained_student', type=int, default=0, help="use pretrained student model for KD")
    parser.add_argument('--resume', type=int, default=0, help="resume training")
    args_ = parser.parse_args()
    main(args_)
