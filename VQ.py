'''
Cluster method
author: Leon, date: 2017.12.04
'''
'''
Data typy:
    m×n 2D array means m vectors in a bach, each vector has n elements
   [[x11,x12...x1n], -> vector 1
    [x21,x22...x2n], -> vector 2
         :                :
    [x1m,x1m...x1m]] -> vector m
Element operation:
    e_func(m×n x, float y)= m×n array,  such as e_max(x,y)[i,j]= max(x[i,j],y)
    e_func(m×n x, m×n y)= m×n array,  such as e_max(x,y)[i,j]= max(x[i,j],y[i,j])
Vector operation:
    v_func(m×n x)= m×? array,  such as v_l2norm(x,y)[i,1]= l2norm(x[i,:])
    v_func(m×n x, k×n y)= m×k array,  such as v_inner(x,y)= x·y'
Batch operation:
    b_func(m×n x)= 1×n array,  such as b_norm(x)[1,i]= norm(x[:,i])

Data typy in model:
Without bias:inputs·weights'=outputs
   Data:           inputs                weights              outputs
   Format:    Batchs×Inputs(B-I)   Outputs×Inputs(O-I)   Batchs×Outputs(B-O)
                 [[2,1,0],             [[1.0,0.0,0.0],       [[2.0,1.0],
   Demo:          [1,2,1],              [0.0,1.0,0.0]]        [1.0,2.0],
                     :                                           :
                  [3,0,3]]                                    [3.0,0.0]]
Or include bias:input_ex·weights_ex'=outputs_ex
   Data:           inputs                weights                     outputs
   Format:  Batchs×(Inputs+1)(B-I1) Outputs×(Inputs+1)(O-I1)  Batchs×Outputs(B-O)
              [[2,1,0,const=1],         [1.0,0.0,0.0,b0],        [[2.0+b0,1.0+b1],
   Demo:       [1,2,1,const=1],         [0.0,1.0,0.0,b1]]         [1.0+b0,2.0+b1],
                     :                                                  :
               [3,0,3,const=1]]                                   [3.0+b0,0.0+b1]]
'''

import numpy as np

'''
Vector Operations
'''
def v_euclid(x,y):
    '''
    Euclidean distance of vectors,
    '''
    x_y_mat = np.zeros((len(x),len(y)))
    for j in range(len(x)):
        for i in range(len(y)):
            x_y_mat[j,i] = np.linalg.norm(x[j] - y[i], ord=2, axis=0, keepdims=True)
    return x_y_mat


def v_inner(x,y):
    '''
    Inner product distance of vectors,
    '''
    return np.dot(x,np.transpose(y))


def v_cosin(x,y):
    '''
    Cosin distance of vectors,
    '''
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, ord=2, axis=1, keepdims=True)
    x = np.divide(x,x_norm)
    y = np.divide(y,y_norm)
    return np.dot(x,np.transpose(y))


def v_l2norm(x):
    '''
    L2 normalize of vectors,
    '''
    x_l2norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    return np.divide(x,x_l2norm)

'''
Element Operations
'''
def e_liner(x):
    return x


def e_relu(x):
    return np.maximum(x,0)


def e_compete(x):
    x_c = np.zeros_like(x)
    for j,xx in enumerate(x):
        x_c[j,np.argmax(xx)] = 1.0
    return x_c


def e_sigmoid(x):
    return np.reciprocal(np.add(np.exp(np.negative(x)),1))


def e_tanh(x):
    return np.tanh(x)


def e_softplus(x):
    return np.log(np.add(1, np.exp(x)))


def e_gauss(x):
    return np.exp(np.negative(np.square(x)))


def v_softmax(x):
    x_y_mat = np.zeros((len(x),len(y)))
    for j in range(len(x)):
        x_soft = np.exp(x[j])
        x[j] = np.divide(x_soft,np.sum(x_soft))
    return x


class k_means:
    '''
    k_means algorithm,
    '''
    def __init__(self,w_list):
        self.w_array = np.array(w_list)
        self.k = len(self.w_array)
        print("k_means __init__("+ str(self.k) +"-means)")

    def predict(self,x_list,dist_funx):
        c_list=[]
        for x in x_list:
            c_x = np.argmin(dist_funx([x], self.w_array))
            c_list.append(c_x)
        return np.array(c_list)

    def update(self,x_list,dist_funx):
        c_list=[]
        for x in x_list:
            print(dist_funx([x], self.w_array))
            c_x = np.argmin(dist_funx([x], self.w_array))
            c_list.append(c_x)
            w_array = self.w_array * 0
        a_num = np.zeros(self.k)
        for i,c in enumerate(c_list):
            w_array[c] += x_list[i]
            a_num[c] += 1.0
        for i,a in enumerate(w_array):
            w_array[i] /= a_num[i]
        up_dist = np.linalg.norm(self.w_array - w_array)
        isend = (up_dist<=0.0000000000001)
        self.w_array = w_array
        return isend,w_array


def grid_2d(height,width):
    '''
    Make 2d grid,
    '''
    yx=[]
    for h in range(height):
        for w in range(width):
            yx.push([h,w])
    yx = np.array(yx)
    dist=[]
    for i in range(len(yx)):
        dist_i=[]
        for j in range(len(yx)):
            dist_i.append(euc_dist(yx[i],yx[j],ax=0))
        dist.append(dist_i)
    return yx,dist


class sofm:
    '''
    sofm network,
    '''
    def __init__(self,n_in,n_out,p_out,d_out,v_func):
        '''
        p_out,d_out = grid_2d(height,width)
        sf = sofm(n_in,height*width,p_out,d_out)
        '''
        self.n_in = n_in
        self.n_out = n_out
        mu, sigma = 0, 0.1
        self.w = np.random.normal(mu, sigma, [n_in,n_out])
        # self.wb = np.vstack((np.random.normal(mu, sigma, [n_in,n_out]), np.zeros((1,n_out))))
        # self.w = self.wb[0:self.wb.shape[0] - 1, :]
        # self.b = self.wb[self.wb.shape[0] - 1, :]
        self.p_out = p_out
        self.d_out = d_out
        self.vec_func = v_func
        self.net_func = e_liner
        print("sofm __init__()")

    def predict(self, x_list):
        x_list = np.hstack((np.array(x_list), np.ones((len(x_list),1))))
        self.net_func(self.vec_func(x_list,self.w))
        return x_list

    def update(self, x_list, ratio):
        x_list = np.hstack((np.array(x_list), np.ones((len(x_list),1))))
        self.net_func(self.vec_func(x_list,self.w))
        return x_list

