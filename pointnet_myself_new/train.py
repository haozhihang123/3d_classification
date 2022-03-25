 #coding=utf-8
"""
Author: haozh
Date: 2022-01
"""

# 导入所需要的库
import torch
import logging
from tqdm import tqdm
from pathlib import Path
from ipdb import set_trace
from torch import nn
import numpy as np
import importlib
from argparse import ArgumentParser
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from models.pointnet import PointNet
import torch.nn.functional as F

# 定义超参数函数
def parse_args():
    parser = ArgumentParser('training Pointnet!')
    parser.add_argument('--use_gpu', default=True, help='gpu mold')
    parser.add_argument('--gpu_id', default='0', help='gpu id')
    parser.add_argument('--batch_size', default=32, help='batch size in training')
    parser.add_argument('--model_name', default='pointnet_cls', help='3d model name')
    parser.add_argument('--category_num', default=40, help='3d dataset category')
    parser.add_argument('--epoch_num', default=200, help='numbers of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, help='learning rate in training')
    parser.add_argument('--log_dir', default='./log/', help='dir of log')
    parser.add_argument('--data_dir', default='/nfs/project/myself/modelnet40_normal_resampled', help='dir of data')
    parser.add_argument('--optimizer', default='Adam', help='optimizer')
    return parser.parse_args()

# 定义准备工作的函数
def preparation(args):
    # 1. 文件目录准备
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)

    # 2. 日志文件
    logging.basicConfig(
        level=logging.INFO,    #控制台打印的日志级别
        filename=args.log_dir + 'model_train.log',
        filemode='w',           ##模式，a：追加模型 w:写模式
        format='%(asctime)s - %(levelname)s:\t%(message)s')   #日志格式

# 定义计算损失
def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

# 主函数
def main(args):
    # 准备工作
    preparation(args)
    if args.use_gpu:
        device = torch.device('cuda')

    # 数据构建
    train_dataset = ModelNetDataLoader(root=args.data_dir, split='train')
    test_dataset = ModelNetDataLoader(root=args.data_dir, split='test')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    # 模型构建
    pointnet = PointNet().to(device)
    pointnet.apply(inplace_relu)
    criterion = get_loss()
    criterion_cls = nn.CrossEntropyLoss().to(device)
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            pointnet.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.SGD(pointnet.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # 开始训练
    for epoch in range(args.epoch_num):
        logging.info('training epoch:%d' % epoch)
        pointnet.train()
        acc = 0.0 
        mean_correct = []
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader),total=len(trainDataLoader)):
            if args.use_gpu:
                points, target = points.to(device), target.to(device)
            # 前向传播
            set_trace()
            points = points.transpose(2, 1)
            pred, trans_feat = pointnet(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]
           
            # 反向传播
            optimizer.zero_grad()
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward() 
            optimizer.step()
            if batch_id % 10 == 0:
                p_info = 'epoch:{} batch:{}/{}  loss:{}     acc:{}'.format(epoch, batch_id, len(trainDataLoader), loss, np.mean(mean_correct[-10:]))
                print(p_info)
                logging.info(p_info)
                mean_correct = []

        # 每个epoch验证一次
        logging.info('eval epoch:%d' % epoch)
        pointnet.eval()
        with torch.no_grad():
            print(len(testDataLoader))
            acc = 0.0 
            for batch_id, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                if args.use_gpu:
                    points, target = points.to(device), target.to(device)
                # 前向传播
                points = points.transpose(2, 1)
                pred,_ = pointnet(points)
                pred_choice = pred.data.max(1)[1]
                acc += torch.eq(pred_choice, target.view(-1).to(device)).sum().item()
            val_accurate = acc / (len(testDataLoader)*args.batch_size)
            logging.info('val_acc:%f' % (val_accurate))
            print('val_acc:',val_accurate)
        
        # 保存模型
        state = {
            'acc': val_accurate,
            'model_state_dict': pointnet,
            'optimizer_state_dict': optimizer,
        }
        torch.save(state, 'log/epoch_' + str(epoch) + '_' + str(val_accurate)[:5] + '.pth')

            


if __name__ == '__main__':
    args = parse_args()
    main(args)