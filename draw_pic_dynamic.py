import numpy as np
import queue
import pandas as pd
import math
import Gaussian_progress_dynamic as gp
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime


x = []
y = []
t = []
x_s = []
y_s = []
distance = []
v_s = 0.1
test_d1 = np.arange(-1.5, 1.5, 0.1)
test_d2 = np.arange(-1.5, 1.5, 0.1)
mu = None


def init_data():
    global x, y, t, x_s, y_s
    data = pd.read_csv("optimized_simulation_13.csv", header = None)
    # print(data)
    data_1 = data.iloc[:,0]
    coordinate_list = np.array(data_1)
    for a in coordinate_list:
        str_list = a.split('[')[1]
        str_list = str_list.split(']')[0]
        coor = str_list.split(',')
        x.append(float(coor[0]))
        y.append(float(coor[1]))

    data_2 = data.iloc[:,1]
    t_list = np.array(data_2)
    for a in t_list:
        t.append(float(a))
        v_t = -2.5 + v_s * a
        x_s.append(v_t)
        y_s.append(v_t)

    for i in range(len(x)):
        distance2 = (x[i] - x_s[i])**2 + (y[i] - y_s[i])**2
        distance.append(math.sqrt(distance2))

def main():
    init_data()
    # 画散点图
    fig = plt.figure(1)
    # for i in range(len(x)):
    #     plt.plot(x[i], y[i], alpha=0.5, marker='o')
    #     plt.scatter()
    plt.scatter(x, y, alpha=0.5, marker='o')
    plt.plot(x, y, c='black')
    plt.scatter(x_s, y_s, c='green', alpha=0.5, marker='o')
    # plt.plot(x, y, linewidth = 4)
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    plt.title('test')
    # 设置横轴的上下限值
    plt.xlim(-2.5, 2.5)
    # 设置纵轴的上下限值
    plt.ylim(-2.5, 2.5)
    # 设置横轴精准刻度
    plt.xticks(np.arange(-2.5, 2.5, step=0.5))
    # 设置纵轴精准刻度
    plt.yticks(np.arange(-2.5, 2.5, step=0.5))
    plt.legend(['tracking point','source point'], loc=2, fontsize = 10)
    '''
    plt.plot(t, distance, linewidth=4)
    plt.title("distance of source and tracking - t", fontsize=20)
    plt.xlabel("t", fontsize=12)
    plt.ylabel("distance", fontsize=12)
    plt.tick_params(axis='both', labelsize=10)
    '''
    plt.show()


if __name__ == '__main__':
    main()