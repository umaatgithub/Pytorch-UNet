import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet
from utils.dataset import BasicDataset
from utils.load_config import data

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

dir_checkpoint = data['training']['checkpoint']['path']


def train_net(net,
              optimizer,
              lr,
              loss_fn,
              epochs,
              batch_size,
              cp_name,
              device):

    if data['training']['preprocess']['flag'] :
        dataset = BasicDataset(data['training']['image']['path'], data['training']['mask']['path'], 1.0)
    else :
        dataset = BasicDataset(data['training']['image']['pre_path'], data['training']['mask']['pre_path'], 1.0)
    n_val = int(len(dataset) * data['training']['validation']['percentage']/100)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'Optim_{optimizer}_LR_{lr}_Loss_{loss_fn}_Epochs_{epochs}_BS_{batch_size}')
    global_step = 0
    if optimizer == 'adam':
        optimizer_fn = optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer_fn = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_fn, 'min' if net.n_classes > 1 else 'max',
                                                         patience=2)
    if net.n_classes > 1:
        if loss_fn == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    #    criterion = nn.BCELoss()
    cp_path = data['training']['checkpoint']['path']+cp_name+'/'

    logging.info(f'''Starting training:
        Optimizer:       {optimizer}
        Learning rate:   {lr}
        Loss function:   {loss_fn}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoint path: {cp_path}
        Device:          {device.type}
    ''')

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks.squeeze(dim=1))
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer_fn.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer_fn.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (len(dataset) // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    #scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer_fn.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/validation', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/validation', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)

        if data['training']['checkpoint']['flag']:
            try:
                os.makedirs(cp_path)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       cp_path + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-o', '--optim', type=str, default='adam',
                        help='Optimizer', dest='optimizer')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-l', '--loss-fn', type=str, default='cross_entropy',
                        help='Loss function', dest='loss_fn')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-cpn', '--checkpoint-name', type=str, default='default',
                        help='Checkpoint name', dest='cp_name')

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    in_channels = data['network']['input']['channels']
    out_channels = data['network']['output']['channels']
    net = UNet(in_channels, out_channels, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if  data['training']['pretrained']['flag']:
        net.load_state_dict(
            torch.load(data['training']['pretrained']['path'], map_location=device)
        )
        logging.info(f"Model loaded from {data['training']['pretrained']['path']}")

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  optimizer=args.optimizer,
                  lr=args.lr,
                  loss_fn= args.loss_fn,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  cp_name=args.cp_name,
                  device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
