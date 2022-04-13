 #coding=utf-8
"""
Author: haozh
Date: 2022-01
"""

# 导入所需要的库
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from ipdb import set_trace
from torch import nn
import numpy as np
import importlib
from argparse import ArgumentParser
import torch.nn.functional as F

# 模型导入
from models.utils.loss import *
from models.utils.ModelNetDataLoader import ModelNetDataLoader
from models import cls_models

# 定义超参数函数
def parse_args():
    parser = ArgumentParser('training Pointnet!')
    parser.add_argument('--use_gpu', default=True, help='gpu mold')
    parser.add_argument('--gpu_id', default='0', help='gpu id')
    parser.add_argument('--batch_size', default=16, help='batch size in training')
    parser.add_argument('--num_workers', default=16, help='num_workers in training')
    parser.add_argument('--model_name', default='PointNet_plus_mrg', help='3d model name')
    parser.add_argument('--category_num', default=40, help='3d dataset category')
    parser.add_argument('--epoch_num', default=100, help='numbers of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, help='learning rate in training')
    parser.add_argument('--log_dir', default='/nfs/project/myself/pointnet_myself_new_v2/log/PointNet_plus_mrg/', help='dir of log')
    parser.add_argument('--data_dir', default='/nfs/project/myself/modelnet40_normal_resampled', help='dir of data')
    parser.add_argument('--optimizer', default='Adam', help='optimizer')
    return parser.parse_args()



# 主函数
def main(args):
    # log记录
    writer = SummaryWriter(args.log_dir + 'logs')
    val_max = 0.0

    # 准备工作
    if args.use_gpu:
        device = torch.device('cuda')

    # 数据构建
    train_dataset = ModelNetDataLoader(root=args.data_dir, split='train')
    test_dataset = ModelNetDataLoader(root=args.data_dir, split='test')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 模型构建
    model = cls_models.__dict__[args.model_name]().to(device)
    criterion = cls_loss(model_name=args.model_name)
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # 开始训练
    for epoch in range(args.epoch_num):
        model.train()
        acc = 0.0 
        mean_correct = []
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader),total=len(trainDataLoader)):
            if args.use_gpu:
                points, target = points.to(device), target.to(device)
            # 前向传播
            points = points.transpose(2, 1)
            pred, trans_feat = model(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]
           
            # 反向传播
            optimizer.zero_grad()
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward() 
            optimizer.step()
            if batch_id % 10 == 0:
                acc = np.mean(mean_correct[-10:])
                p_info = 'epoch:{} batch:{}/{}  loss:{}     acc:{}'.format(epoch, batch_id, len(trainDataLoader), loss, acc)
                tqdm.write(p_info)
                writer.add_scalar("train_loss",loss,epoch*len(trainDataLoader)+batch_id)
                writer.add_scalar("train_acc",acc,epoch*len(trainDataLoader)+batch_id)
                mean_correct = []
                

        # 每个epoch验证一次
        model.eval()
        with torch.no_grad():
            print(len(testDataLoader))
            acc = 0.0 
            for batch_id, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                if args.use_gpu:
                    points, target = points.to(device), target.to(device)
                # 前向传播
                points = points.transpose(2, 1)
                pred, trans_feat = model(points)
                loss = criterion(pred, target.long(), trans_feat)
                pred_choice = pred.data.max(1)[1]
                acc += torch.eq(pred_choice, target.view(-1).to(device)).sum().item()
            val_accurate = acc / (len(testDataLoader)*args.batch_size)
            print('val_loss:{}  val_acc:{}'.format(loss,val_accurate))
            writer.add_scalar("val_loss",loss,epoch)
            writer.add_scalar("val_acc",val_accurate,epoch)
        # 保存模型
        if val_max < val_accurate:
            val_max = val_accurate
            state = {
                'acc': val_accurate,
                'model': model,
                'optimizer': optimizer,
            }
            torch.save(state, args.log_dir + 'epoch_' + str(epoch) + '_' + str(val_accurate)[:5] + '.pth')
    writer.close()
            


if __name__ == '__main__':
    args = parse_args()
    main(args)