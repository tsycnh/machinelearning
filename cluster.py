'''
cluster method
'''
import numpy as np

def euc_dist(x,y,ax=1):
    '''
    Euclidean distance of vectors,
    see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
    '''
    x = np.array(x)
    y = np.array(y)
    return np.linalg.norm(x-y,ord=2,axis=ax,keepdims=True)

class k_means:
    '''
    k_means method,
    '''
    def __init__(self,a_list):
        self.a_array = np.array(a_list)
        #self.a_list = np.array(a_list)
        self.k = len(self.a_array)
        print("k_means __init__("+ str(self.k) +"-means)")

    def predict(self,x_list,dist_funx):
        c_list=[]
        for x in x_list:
            c_x = np.argmin(dist_funx([x],self.a_array))
            c_list.append(c_x)
        return np.array(c_list)

    def update(self,x_list,dist_funx):
        c_list=[]
        for x in x_list:
            c_x = np.argmin(dist_funx([x],self.a_array))
            c_list.append(c_x)
            a_array = self.a_array*0
        a_num = np.zeros(self.k)
        for i,c in enumerate(c_list):
            a_array[c] += x_list[i]
            a_num[c] += 1.0
        for i,a in enumerate(a_array):
            a_array[i] /= a_num[i]
        up_dist = np.linalg.norm(self.a_array - a_array)
        isend = (up_dist<=0.0000000000001)
        self.a_array = a_array
        return isend,a_array
