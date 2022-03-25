import torch 
import numpy as np
from ipdb import set_trace
import numpy as np
import torch.nn.functional as F
import os
from tqdm import tqdm

# 加载标签种类
data_root = '/nfs/project/myself/modelnet40_normal_resampled/'
catfile = data_root + 'modelnet40_shape_names.txt'
cat = [line.rstrip() for line in open(catfile)]

# 加载模型
model_file = './log/epoch_0.883.pth'
model = torch.load(model_file)['model_state_dict'].cuda().eval()  

test_path = data_root + 'modelnet40_test.txt'
test_files = cat = [line.rstrip() for line in open(test_path)]

# 加载带预测点云数据
point_file = data_root + 'airplane/airplane_0004.txt'
point_set = np.loadtxt(point_file, delimiter=',').astype(np.float32)[0:1024, :]
point_set = torch.unsqueeze(torch.tensor(point_set),0).transpose(2, 1).cuda()

# 前向传播
pred,_ = model(point_set)
pred_choice = pred.data.max(1)[1]
score = torch.squeeze(F.softmax(pred, dim=1))[pred_choice].item()
sign = cat[pred_choice.item()]

print('file:{}   cls:{}   score:{}'.format(point_file.split('/')[-1],sign,score))
