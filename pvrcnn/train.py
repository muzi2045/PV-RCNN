import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import multiprocessing
import time
import sys

from tensorboardX import SummaryWriter
from pvrcnn.detector import ProposalLoss, PV_RCNN, Second
from pvrcnn.core import cfg, TrainPreprocessor, VisdomLinePlotter
from pvrcnn.dataset import KittiDatasetTrain
from pvrcnn.dataset import UDIDatasetTrain
from pvrcnn.dataset import NuscenesDatasetTrain


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def build_train_dataloader(cfg, preprocessor):
    dataloader = DataLoader(
        NuscenesDatasetTrain(cfg),
        collate_fn=preprocessor.collate,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=2,
    )
    return dataloader


def save_cpkt(model, optimizer, epoch, meta=None):
    fpath = f'./ckpts/epoch_{epoch}.pth'
    ckpt = dict(
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
        epoch=epoch,
        meta=meta,
    )
    os.makedirs('./ckpts', exist_ok=True)
    torch.save(ckpt, fpath)


def load_ckpt(fpath, model, optimizer):
    if not osp.isfile(fpath):
        return 0
    ckpt = torch.load(fpath)
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt['epoch']
    return epoch


def update_plot(losses, prefix):
    for key in ['loss', 'cls_loss', 'reg_loss']:
        print(f'{prefix}_{key}', losses[key].item())
        # plotter.update(f'{prefix}_{key}', losses[key].item())

def init_tensorboardX():
    summary_dir = './summary'
    os.makedirs(summary_dir, exist_ok=True)
    summary_writer = SummaryWriter(str(summary_dir))
    return summary_writer

def update_tensorboardX(writer, losses, predfix, step):
    for key in ['loss', 'cls_loss', 'reg_loss']:
        writer.add_scalar(predfix + "/", losses[key].item() ,step)

def to_device(item):
    keys = ['G_cls', 'G_reg', 'M_cls', 'M_reg', 'points',
        'features', 'coordinates', 'occupancy']
    for key in keys:
        item[key] = item[key].cuda()


def train_model(model, dataloader, optimizer, lr_scheduler, loss_fn, epochs, start_epoch, need_log, saver, model_save_path):
    # model.train()
    summary_writer = init_tensorboardX()
    for epoch in range(start_epoch, epochs):
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}, learning rate {lr}")
        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()
        model.train()
        for step, item in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}')):
            to_device(item)
            optimizer.zero_grad()
            out = model(item)
            losses = loss_fn(out)
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
            optimizer.step()
            lr_scheduler.step()
            if(step % 50) == 0:
                update_tensorboardX(summary_writer, losses, 'step', 5714*epoch+ step)
            if (step % 100) == 0:
                update_plot(losses, 'step')
        if need_log and (epoch % 5 == 0 or epoch == epochs or epoch == 1 or epoch > 20):
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': losses['loss']
            }
            torch.save(save_dict, os.path.join(model_save_path, 'epoch_' + str(epoch) + '.pth'))

    if need_log:
        saver.close()    
    summary_writer.close()


def build_lr_scheduler(optimizer, cfg, start_epoch, N):
    last_epoch = start_epoch * N / cfg.TRAIN.BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.TRAIN.LR, steps_per_epoch=N,
        epochs=cfg.TRAIN.EPOCHS, last_epoch=-1)
    return scheduler


def main(args):
    """TODO: Trainer class to manage objects."""
        
    # cfg.merge_from_file('../configs/Nuscenes/all_class_lite.yaml')
    cfg.merge_from_file(args.config)
    need_log = args.log
    start_epoch = 1

    if need_log:
        logger_root = args.logpath if args.logpath != '' else 'logs'
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        if args.resume == '':
            model_save_path = check_folder(logger_root)
            model_save_path = check_folder(os.path.join(model_save_path, time_stamp))
            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "w")
            saver.write(f"GPU number: {torch.cuda.device_count()}\n")
            saver.flush()

            saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()

        else:
            model_save_path = args.resume
            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "a")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print(f"device number {device_num}")

    model = Second(cfg)
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model = model.to(device)

    loss_fn = ProposalLoss(cfg)
    preprocessor = TrainPreprocessor(cfg)
    dataloader = build_train_dataloader(cfg, preprocessor)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    scheduler = build_lr_scheduler(optimizer, cfg, start_epoch, len(dataloader))
    
    ### resume the model file if training paused.
    if args.resume != '':
        assert os.path.isfile(args.resume), f"the resume file path don't exists...+_+"
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Load model from {args.resume}, at epoch {start_epoch - 1}")
    
    num_epochs = cfg.TRAIN.EPOCHS

    # for epoch in range(start_epoch, num_epochs + 1):
    #     lr = optimizer.param_groups[0]['lr']
    #     print(f"Epoch {epoch}, learning rate {lr}")
    #     if need_log:
    #         saver.write("epoch: {}, lr : {}\t".format(epoch, lr))
    #         saver.flush()
        
    #     scheduler.step()
    #     model.train()

    train_model(model, dataloader, optimizer,
        scheduler, loss_fn, num_epochs, start_epoch, need_log, saver, model_save_path)


if __name__ == '__main__':
    # try:
    #     multiprocessing.set_start_method('spawn')
    # except RuntimeError:
    #     pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', type=str, help='the config path for trained model.')
    parser.add_argument('--resume', default='', type=str, help='the path of model to be resume.')
    parser.add_argument('--log', action='store_true', help='whether to log.')
    parser.add_argument('--logpath', default='', help='The path to the output log file')
    args = parser.parse_args()
    print(f"{args}")
    main(args)
