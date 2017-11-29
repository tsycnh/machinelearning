import matplotlib.pyplot as plt
import numpy as np

def sample_2d(num=1,xlim=[-1, 1],ylim=[-1, 1]):
    fig = plt.figure()
    ax = fig.gca()
    #设置坐标轴范围
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #设置坐标轴名称
    plt.xlabel('x1')
    plt.ylabel('x2')
    #设置坐标轴刻度
    # x_ticks = np.arange(-1, 1.2, 0.2)
    # y_ticks = np.arange(-1, 1.2, 0.2)
    # ax.set_xticks(x_ticks)
    # ax.set_yticks(y_ticks)
    #隐藏坐标轴
    #ax.axis('off')
    #设置坐标轴缩放
    ax.set_aspect('equal')
    ax.grid(color='b', linestyle=':', linewidth=1)
    #plt.grid()
    #取坐标点
    sample=np.array(plt.ginput(num))
    fig.clf()
    plt.close()
    return sample

def plot_2d(sample,xlim=None,ylim=None,fig=None,mark='ro'):
    sample=np.array(sample)
    if xlim is None:
        xlim = [np.amin(sample[:,0]),np.amax(sample[:,0])]
        xrang = xlim[1]-xlim[0]
        xlim = [xlim[0]-xrang/8,xlim[1]+xrang/8]
    if ylim is None:
        ylim = [np.amin(sample[:,1]),np.amax(sample[:,1])]
        yrang = ylim[1]-ylim[0]
        ylim = [ylim[0]-yrang/8,xlim[1]+yrang/8]
    needShow=False
    if fig is None:
        fig = plt.figure()
        needShow=True
    ax = fig.gca()
    #设置坐标轴范围
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #设置坐标轴名称
    plt.xlabel('x1')
    plt.ylabel('x2')
    #设置坐标轴刻度
    # x_ticks = np.arange(-1, 1.2, 0.2)
    # y_ticks = np.arange(-1, 1.2, 0.2)
    # ax.set_xticks(x_ticks)
    # ax.set_yticks(y_ticks)
    #隐藏坐标轴
    #ax.axis('off')
    #设置坐标轴缩放
    ax.set_aspect('equal')
    ax.grid(color='b', linestyle=':', linewidth=1)
    ax.plot(sample[:,0], sample[:,1],mark,markerfacecolor = 'white')
    if needShow:
        plt.show()
    return fig