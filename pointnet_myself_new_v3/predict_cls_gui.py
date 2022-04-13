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
catfile = data_root + '\\modelnet40_shape_names.txt'
cat = [line.rstrip() for line in open(catfile)]

# 加载模型
model_file = data_root + '\\log\\PointNet_plus_msg\\epoch_0_0.745.pth'
model = torch.load(model_file,  map_location=torch.device('cpu'))['model'].eval()  

# 读取点云数据
def read_point_file(filename, Separator):
  data = np.loadtxt(filename,delimiter = Separator)
  point = [data[:,0].tolist(),data[:,1].tolist(),data[:,2].tolist()]
  return point, data
 
#三维离散点图显示点云
def displayPoint(data,title,save_name):
    #散点图参数设置
    fig=plt.figure() 
    ax=Axes3D(fig) 
    ax.set_title(title) 
    ax.scatter3D(data[0], data[2], data[1], c = 'r', marker = '.') 
    ax.set_xlabel('x') 
    ax.set_ylabel('y') 
    ax.set_zlabel('z') 
    plt.savefig(save_name)
    # plt.show()


#初始化GUI界面
top=tk.Tk()
top.geometry('800x600')
top.title('Point Classification Momdenet40')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',25,'bold'))
sign_image = Label(top)

def classify():
    global label_packed
    # 加载带预测点云数据
    point_set = point_all.astype(np.float32)[0:1024, :]
    point_set = torch.unsqueeze(torch.tensor(point_set),0).transpose(2, 1)

    # 前向传播
    pred,_ = model(point_set)
    pred_choice = pred.data.max(1)[1]
    score = torch.squeeze(F.softmax(pred, dim=1))[pred_choice].item()
    sign = cat[pred_choice.item()]
    label.configure(foreground='#011638', text=sign + '(' + str(score)+ ')') 

def show_classify_button():
    classify_b=Button(top,text="Classify point",
   command=lambda: classify(),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',
font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_point():
    try:
        global point_all, point_file_name
        file_path=filedialog.askopenfilename()
        point_file_name = file_path.split('/')[-1]
        point_xyz, point_all = read_point_file(file_path,',')
        displayPoint(point_xyz, file_path.split('/')[-1].split('.')[0], file_path.replace('txt','png'))
        uploaded=image_read.open(file_path.replace('txt','png'))
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button()
        os.remove(file_path.replace('txt','png'))
    except:
        pass
    
upload=Button(top,text="Upload an image",command=upload_point,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Image Classification Momdenet40",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()

