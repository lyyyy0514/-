import numpy as np 
# test
a1=0.9
a2=0.5
a3=0.7
w1=0.8
w2=-0.4
w3=0
b1=1
sum1=a1*w1+a2*w2+a3*w3+b1
print(sum1)
#激活函数
def Relu(x):
    return np.maximum(0,x)
print(Relu(-sum1))
#使用numpy简洁表示
inputs=np.array([a1,a2,a3])
weight=np.array([w1,w2,w3])
sum1=np.dot(inputs,weight)+b1
print(sum1)
#用矩阵表示

weights=np.array([[w1,w2,w3],[0.2,0.4,0.6],[0.5,0.3,0.1]])
sum2=np.dot(inputs,weights)+b1
print(sum2)
sum3=Relu(sum2)
print(sum3)
# 这个时候数据都只是一个数据，经常数据都是batch的形式，输入是一个矩阵，每一行是一个数据
inputs=np.array([[a1,a2,a3],[0.5,0.6,0.7]])
weights=np.array([[w1,w2,w3],[0.2,0.4,0.6],[0.5,0.3,0.1]])
sum4=np.dot(inputs,weights)+b1  
print(sum4)



def create_weights(n_inputs,n_neural):
    return np.random.randn(n_inputs,n_neural)
create_weight=create_weights(3,4)
print(create_weight)
# 偏置也可以不一样，b也可以是一个向量,这里通常涉及广播机制
a1=np.array([[1,2],[3,4]])
a2=np.array([-1,1])
print(a1+a2)#这里就用到了广播机制，相当于复制了一行
a3=np.array([[-1],[1]])
print(a1+a3)
def create_biases(n_neural):
    return np.random.randn(n_neural)
create_biase=create_biases(2)
print(create_biase)
#第一层
weights1=create_weights(2,3)
biases1=create_biases(3)
#第二层
weights2=create_weights(3,4)
biases2=create_biases(4)
#第三层
weights3=create_weights(4,2)
biases3=create_biases(2)
data=np.array([[1,2],[2,3],[3,4],[4,5]])
a1=Relu(data@weights1+biases1)
a2=Relu(a1@weights2+biases2)
a3=Relu(a2@weights3+biases3)
print(a3)
#面向对象的神经网络，写成一个类
class Layer():
    def __init__(self,n_input,n_neural):
        self.weights=np.random.rand(n_input,n_neural)
        self.biases=np.random.rand(n_neural)
    def Relu(self,x):
        return np.maximum(0,x)
    def layer_forward(self,inputs):
        sum=inputs@self.weights+self.biases  
        return Relu(sum)
layer1=Layer(2,3)
layer2=Layer(3,4)
layer3=Layer(4,2)
output1=layer1.layer_forward(inputs=data)
output2=layer2.layer_forward(inputs=output1)
output3=layer3.layer_forward(inputs=output2)
print(output3)
#定义神经网络的类
class network():
    def __init__(self,network_shape):
        self.network_shape=network_shape
        for i in range(len(network_shape)-1):
            