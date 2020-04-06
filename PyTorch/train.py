import time
import argparse
import datetime
import numpy as np
import os 
import pathlib
import shutil

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
from torch.utils.tensorboard import SummaryWriter

from model import Model, MyDataParallel
from loss import ssim
from data import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize, compute_errors

def save_model(state, is_best, dir, filename='checkpoint.pth.tar'):
    torch.save(state, dir + filename)
    if is_best:
        print("\tSaving new best model", flush=True)
        shutil.copyfile(dir + filename, dir + 'model_best.pth.tar')

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--datadir', default='E:/Source/R4DL/data/', type=str, help='Dataset directory.')
    parser.add_argument('--datacsv', default='data/nyu2_train.csv', type=str, help='Dataset csv file.')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    parser.add_argument('--name', type=str, default='densedepth_nyu', help='A name to attach to the training session')
    args, _ = parser.parse_known_args()

    if ('PT_DATA_DIR' in os.environ):
        args.datadir = os.environ['PT_DATA_DIR'] + '/'

    model = Model()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if (torch.cuda.is_available() and torch.cuda.device_count() > 1):
      print("Let's use", torch.cuda.device_count(), "GPUs!", flush=True)
      model = MyDataParallel(model)

    # Create model
    model = model.to(device)
    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam( model.parameters(), args.lr )
    batch_size = args.bs
    prefix = 'densenet_' + str(batch_size)

    # Load data
    train_loader, val_loader = getTrainingTestingData(batch_size, args.datadir, args.datacsv)

    # Training session details
    runID = str(int(time.time())) + '-n' + str(len(train_loader)) + '-e' + str(args.epochs) + '-bs' + str(args.bs) + '-lr' + str(args.lr) + '-' + args.name
    outputPath = args.datadir + 'models/'
    runPath = outputPath + runID + '/'
    pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
    print('Output: ' + runPath)

    # Logging
    writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)

    # Loss
    l1_criterion = nn.L1Loss()

    best_loss = 1e10

    # Start training...
    for epoch in range(args.epochs):

        # train for one epoch
        train_loss = trainAndVal(train_loader, model, l1_criterion, optimizer)
        
        writer.add_scalar("train_loss", train_loss, global_step=epoch)
  
        # evaluate on validation set
        val_loss = trainAndVal(val_loader, model, l1_criterion)

        writer.add_scalar("val_loss", val_loss, global_step=epoch)    

        # remember best loss and save checkpoint
        is_best = val_loss < best_loss

        print('new %f best %f loss' % (val_loss, best_loss), flush=True)

        writer.add_scalar("best_loss", best_loss, global_step=epoch)               
        writer.close()

        best_loss = min(val_loss, best_loss)

        save_model({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),                                    
        }, is_best, runPath)

def trainAndVal(loader, model, l1_criterion, optimizer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    rsmes = AverageMeter()

    if (optimizer):
        # switch to train mode
        model.train()
        print('Train', flush=True)
    else:
        # switch to evaluate mode
        model.eval()
        print('Val', flush=True)
        
    N = len(loader)

    end = time.time()
    start = end

    if (optimizer is None):
        predictions = []
        testSetDepths = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # process epoch
    for i, sample_batched in enumerate(loader):

        # Prepare sample and target
        image = sample_batched['image'].to(device)
        depth = sample_batched['depth'].to(device)

        # Normalize depth
        depth_n = DepthNorm( depth )

        # Predict
        output = model(image)

        # Compute the loss
        l_depth = l1_criterion(output, depth_n)
        l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)

        loss = (1.0 * l_ssim) + (0.1 * l_depth)

        # measure accuracy and record loss
        losses.update(loss.data, image.size(0))

        rmse = (depth_n.data.cpu() - output.data.cpu()) ** 2
        rmse = np.sqrt(rmse.mean())
        rsmes.update(rmse, image.size(0))

        if (optimizer):
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        eta = str(datetime.timedelta(seconds=int(batch_time.avg*(N - i))))
        total = str(datetime.timedelta(seconds=int((time.time() - start) + batch_time.avg*(N - i))))

        minDepth=10
        maxDepth = 1000

        if (optimizer is None):

            predictions.append(output.squeeze().data.cpu())
            testSetDepths.append(depth_n.squeeze().data.cpu())

        if i % 5 == 0:
            p = 100*i/N
            bar = "[%-10s] %d%%" % ('='*int(p*10/100) + '.'*(10-int(p*10/100)), p)
            print('[{0}/{1}] {2} - '
                'Batch Time: {batch_time.val:.2f} ({batch_time.avg:.2f}) '
                'ETA: {eta}/{total} '
                'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                'RSME: {rsme.val:.3f} ({rsme.avg:.3f})'
                .format(i, N, bar, batch_time=batch_time, eta=eta, total= total, loss=losses, rsme=rsmes), flush=True)

            break

    if (optimizer is None):
        predictions = np.vstack(predictions)
        testSetDepths = np.vstack(testSetDepths)

        e = compute_errors(predictions, testSetDepths)

        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))

    return losses.avg

if __name__ == '__main__':
    main()
