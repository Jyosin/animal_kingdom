import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
import random

#接下来这个部分我要做三种运动轨迹

def trajectory(t):

    x = 100 * np.cos(2 * np.pi * random.random() * t) + random.randint(100,500)
    y = 100 * np.sin(2 * np.pi * random.random() * t) + random.randint(100,500)

    return x,y

#在这一步，根据所定的标签数量n，生成[1*n] [2*视频帧*n] 的list
def make_lable_list(duration,fps,n,l):

    total_frames = duration * fps
    t = np.linspace(0, duration, total_frames)
    
    data_list = []
    label_list = []
    data_list_temp = []

    for label in range(n) :
        for t_now in t:
            data_list_temp.append(trajectory(t_now))
            if len(data_list_temp) >= l:
                data_list.append(data_list_temp[-l:])
                label_list.append(label)
    

    a_data_list = np.array(data_list)
    a_label_list = np.array(label_list)

    data = torch.tensor(a_data_list, dtype=torch.float32)
    labels = torch.tensor(a_label_list, dtype=torch.long)

    torch.save(data, 'data/shape_paths_data.pt')
    torch.save(labels, 'data/shape_labels.pt')

make_lable_list(20,20,2,20)









