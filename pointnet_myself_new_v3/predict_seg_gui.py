import os
import torch 
import numpy as np
from tqdm import tqdm
from ipdb import set_trace
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image as image_read
from PIL import ImageTk

# 加载标签种类
data_root = os.getcwd()

# 加载模型
model_file = data_root + '\\log\\PointNet_plus_msg_seg\\epoch_91_0.929.pth'
model = torch.load(model_file,  map_location=torch.device('cpu'))['model'].eval()  

# 读取点云数据
def read_point_file(filename):
    data = np.loadtxt(filename,delimiter = ' ')
    classes = list(set(data[:,-1]))
    points = []
    for clsass in classes:
        class_data = data[data[:,-1] == clsass]
        points.append([class_data[:,0].tolist(),class_data[:,1].tolist(),class_data[:,2].tolist()])
    points_all = [data[:,0].tolist(),data[:,1].tolist(),data[:,2].tolist()]
    data_all = {'points':points, 'points_all':points_all, 'points_forword':data[:,:6]}
    return data_all
 
#三维离散点图显示点云
def displayPoint(data,save_name,flag):
    colors = ['#8A2BE2','#A52A2A','#DEB887','#5F9EA0','#7FFF00','#D2691E','#FF7F50','#6495ED','#FFF8DC','#DC143C',
              '#00FFFF','#00008B','#008B8B','#B8860B','#A9A9A9','#006400','#BDB76B','#8B008B','#556B2F','#FF8C00',
              '#9932CC','#8B0000','#E9967A','#8FBC8F','#483D8B','#2F4F4F','#00CED1','#9400D3','#FF1493','#00BFFF',
              '#00FFFF','#7FFFD4','#F0FFFF','#F5F5DC','#FFE4C4','#000000','#FFEBCD','#0000FF','#F0F8FF','#FAEBD7']
    #散点图参数设置
    fig=plt.figure() 
    ax=Axes3D(fig) 
    if flag != 'all':
        for i in range(len(data)):
            ax.scatter3D(data[i][0], data[i][2], data[i][1], c = colors[i], marker = '.') 
    else:
        ax.scatter3D(data[0], data[2], data[1], c = 'gray', marker = '.') 
    ax.set_xlabel('x') 
    ax.set_ylabel('y') 
    ax.set_zlabel('z') 
    plt.savefig(save_name)
    # plt.show()
    


#初始化GUI界面
top=tk.Tk()
top.geometry('800x600')
top.title('Point part Classification shapenet')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',25,'bold'))
sign_image = Label(top)

def classify():
    global label_packed
    # 加载带预测点云数据
    # set_trace()
    point_all = data_all['points_forword']
    point_set = point_all.astype(np.float32)
    point_set = torch.unsqueeze(torch.tensor(point_set),0).transpose(2, 1)

    # 前向传播
    pred,_ = model(point_set)
    pred_choice = pred.data.max(2)[1]
    pred_choice = torch.squeeze(pred_choice)
    np_pred_choice = np.float64(pred_choice.numpy())
    classes = np.unique(np_pred_choice)
    points = []
    for i in range(len(classes)):
        class_data = point_all[np_pred_choice == classes[i]]
        points.append([class_data[:,0].tolist(),class_data[:,1].tolist(),class_data[:,2].tolist()])
    displayPoint(points, point_file_name.replace('txt','png'),flag='part')
    uploaded=image_read.open(point_file_name.replace('txt','png'))
    uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
    im=ImageTk.PhotoImage(uploaded)
    sign_image.configure(image=im)
    sign_image.image=im
    label.configure(text='')
    os.remove(point_file_name.replace('txt','png'))

def show_classify_button():
    classify_b=Button(top,text="Classify point",
   command=lambda: classify(),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',
font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_point():
    # try:
    global data_all, point_file_name
    file_path=filedialog.askopenfilename()
    point_file_name = file_path
    data_all = read_point_file(file_path)
    displayPoint(data_all['points_all'], file_path.replace('txt','png'),flag='all')
    uploaded=image_read.open(file_path.replace('txt','png'))
    uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
    im=ImageTk.PhotoImage(uploaded)
    sign_image.configure(image=im)
    sign_image.image=im
    label.configure(text='')
    show_classify_button()
    os.remove(file_path.replace('txt','png'))
    # except:
    #     pass
    
upload=Button(top,text="Upload an image",command=upload_point,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="point part Classification shapenet",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()

