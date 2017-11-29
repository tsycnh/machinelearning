# Load sample datas
import numpy as np
import random
import cluster
import matplotlib.pyplot as plt
import gui
k = 5
n = 5
x_list=[]
x_array = None
a_array = None
figure = plt.figure()
marks_x = ['ko','kv','ks','kp','kh','k*']
marks_a = ['ro','rv','rs','rp','rh','r*']
x_lim=[-1, 1]
y_lim=[-1, 1]
for i in range(k):
    xi = gui.sample_2d(num=n, xlim=x_lim, ylim=y_lim)
    x_list.append(xi)
    figure=gui.plot_2d(sample=xi,xlim=x_lim, ylim=y_lim,fig=figure,mark=marks_x[i])
for xi in x_list:
    if x_array is None:
        x_array = xi
    else:
        x_array = np.vstack((x_array[:,:], xi))
a_index = random.sample(range(n*k), k)
a_array = x_array[a_index]
for i,a in enumerate(a_array):
    figure = gui.plot_2d(sample=[a], xlim=x_lim, ylim=y_lim, fig=figure, mark=marks_a[i])
plt.show()
plt.close('all')

# Run algorithm
k_means = cluster.k_means(a_array)
# k_means.predict(x, cluster.euc_dist)
for i in range(100):
    isend, _ = k_means.update(x_array, cluster.euc_dist)
    if isend:
        print("Cluster convergence")
        break
    figure = plt.figure()
    for i,x in enumerate(x_list):
        figure=gui.plot_2d(sample=x,xlim=x_lim, ylim=y_lim,fig=figure,mark=marks_x[i])
    for i,s in enumerate(k_means.a_array):
        figure = gui.plot_2d(sample=[s], xlim=x_lim, ylim=y_lim, fig=figure, mark=marks_a[i])
    plt.show()