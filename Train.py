import os
import argparse

from tqdm import tqdm
import pandas as pd
import joblib


from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn

import torch.optim as optim

from Networks.net import MODEL as net


from losses import CharbonnierLoss_ir, CharbonnierLoss_vi,ssim_loss_vi,ssim_loss_ir


device = torch.device('cuda:0')
use_gpu = torch.cuda.is_available()

if use_gpu:
    print('GPU Mode Acitavted')
else:
    print('CPU Mode Acitavted')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='...', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float)    # 5e-4
    parser.add_argument('--weight', default=[1,1,4,4], type=float)

    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--weight-decay', default=1e-5, type=float)
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    args = parser.parse_args()

    return args




class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader_ir, model, criterion_ir_Charbonnier, criterion_vi_Charbonnier,criterion_ssim_ir,criterion_ssim_vi,optimizer):
    losses = AverageMeter()
    losses_ir_Charbonnier = AverageMeter()
    losses_vi_Charbonnier = AverageMeter()
    losses_ssim_ir = AverageMeter()
    losses_ssim_vi= AverageMeter()
    weight = args.weight
    model.train()

    for i, (input,ir,vi)  in tqdm(enumerate(train_loader_ir), total=len(train_loader_ir)):

        if use_gpu:
            input = input.cuda()

            ir=ir.cuda()
            vi=vi.cuda()


        else:
            input = input
            ir=ir
            vi=vi

        out = model(ir,vi)

        loss_ir_Charbonnier = weight[0] * criterion_ir_Charbonnier(out, ir)
        loss_vi_Charbonnier = weight[1] * criterion_vi_Charbonnier(out, vi)
        loss_ssim_ir= weight[2] * criterion_ssim_ir(out,ir)
        loss_ssim_vi = weight[3] * criterion_ssim_vi(out, vi)
        loss = loss_ir_Charbonnier + loss_vi_Charbonnier+loss_ssim_ir+ loss_ssim_vi

        losses.update(loss.item(), input.size(0))
        losses_ir_Charbonnier.update(loss_ir_Charbonnier.item(), input.size(0))
        losses_vi_Charbonnier.update(loss_vi_Charbonnier.item(), input.size(0))
        losses_ssim_ir.update(loss_ssim_ir.item(), input.size(0))
        losses_ssim_vi.update(loss_ssim_vi.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('loss_ir_Charbonnier', losses_ir_Charbonnier.avg),
        ('loss_vi_Charbonnier', losses_vi_Charbonnier.avg),
        ('loss_ssim_ir', losses_ssim_ir.avg),
        ('loss_ssim_vi', losses_ssim_vi.avg),
    ])

    return log



def main():
    args = parse_args()

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    cudnn.benchmark = True

    train_loader_ir = "..."

    train_loader_vi  = ".../"


    model = net(in_channel=1)
    if use_gpu:
        model = model.cuda()
        model.cuda()

    else:
        model = model
    criterion_ir_Charbonnier = CharbonnierLoss_ir
    criterion_vi_Charbonnier = CharbonnierLoss_vi
    criterion_ssim_ir = ssim_loss_ir
    criterion_ssim_vi = ssim_loss_vi
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps)
    log = pd.DataFrame(index=[],
                       columns=['epoch',

                                'loss',
                                'loss_ir_Charbonnier',
                                'loss_vi_Charbonnier',
                                'loss_ssim_ir',
                                'loss_ssim_vi',
                                ])

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch+1, args.epochs))

        train_log = train(args, train_loader_ir,train_loader_vi, model, criterion_ir_Charbonnier, criterion_vi_Charbonnier,criterion_ssim_ir,criterion_ssim_vi, optimizer, epoch)     # 训练集


        print('loss: %.4f - loss_ir_Charbonnier: %.4f - loss_vi_Charbonnier: %.4f - loss_ssim_ir: %.4f - loss_ssim_vi: %.4f '
              % (train_log['loss'],
                 train_log['loss_ir_Charbonnier'],
                 train_log['loss_vi_Charbonnier'],
                 train_log['loss_ssim_ir'],
                 train_log['loss_ssim_vi'],
                 ))

        tmp = pd.Series([
            epoch + 1,

            train_log['loss'],
            train_log['loss_ir_Charbonnier'],
            train_log['loss_vi_Charbonnier'],
            train_log['loss_ssim_ir'],
            train_log['loss_ssim_vi'],

        ], index=['epoch', 'loss', 'loss_ir_Charbonnier', 'loss_vi_Charbonnier', 'loss_ssim_ir', 'loss_ssim_vi'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)
        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), 'models/%s/model_{}.pth'.format(epoch+1) %args.name)


if __name__ == '__main__':
    main()


