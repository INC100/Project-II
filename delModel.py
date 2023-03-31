"""
删除过渡模型
"""

import os
import sys
import shutil

#分别检查Discussion和Models文件夹是否存在
#搜索存在的文件夹下的所有文件
#文件名中“_"及".pth"之间的数字为epoch数
#如果epoch%5!=0，则删除该文件

def delModel():
    if os.path.exists("./Discussion"):
        for file in os.listdir("./Discussion"):
            if file.endswith(".pth"):
                epoch = int(file.split("_")[1].split(".")[0])
                if epoch%5!=0:
                    os.remove("./Discussion/"+file)
    if os.path.exists("./Models"):
        for file in os.listdir("./Models"):
            if file.endswith(".pth"):
                epoch = int(file.split("_")[1].split(".")[0])
                if epoch%5!=0:
                    os.remove("./Models/"+file)

delModel()