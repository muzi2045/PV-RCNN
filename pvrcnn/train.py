import os
import os.path as osp
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import multiprocessing

from tensorboardX import SummaryWriter
from pvrcnn.detector import ProposalLoss, PV_RCNN, Second
from pvrcnn.core import cfg, TrainPreprocessor, VisdomLinePlotter
from pvrcnn.dataset import KittiDatasetTrain
from pvrcnn.dataset import UDIDatasetTrain
from pvrcnn.dataset import NuscenesDatasetTrain


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


def train_model(model, dataloader, optimizer, lr_scheduler, loss_fn, epochs, start_epoch=0):
    model.train()
    summary_writer = init_tensorboardX()
    for epoch in range(start_epoch, epochs):
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
        if (epoch % 3) == 0 or (epoch == epochs - 1):
            save_cpkt(model, optimizer, epoch)
    summary_writer.close()


def build_lr_scheduler(optimizer, cfg, start_epoch, N):
    last_epoch = start_epoch * N / cfg.TRAIN.BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, steps_per_epoch=N,
        epochs=cfg.TRAIN.EPOCHS, last_epoch=-1)
    return scheduler


def main():
    """TODO: Trainer class to manage objects."""
    model = Second(cfg).cuda()
    # model = PV_RCNN(cfg).cuda()
    parameters = model.parameters()
    loss_fn = ProposalLoss(cfg)
    preprocessor = TrainPreprocessor(cfg)
    dataloader = build_train_dataloader(cfg, preprocessor)
    optimizer = torch.optim.Adam(parameters, lr=0.01)
    # start_epoch = 0
    start_epoch = load_ckpt('./ckpts/epoch_10.pth', model, optimizer)
    scheduler = build_lr_scheduler(optimizer, cfg, start_epoch, len(dataloader))
    train_model(model, dataloader, optimizer,
        scheduler, loss_fn, cfg.TRAIN.EPOCHS, start_epoch)


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    # global plotter
    # plotter = VisdomLinePlotter(env='training')
    cfg.merge_from_file('../configs/Nuscenes/all_class_lite.yaml')
    main()
