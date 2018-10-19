# coding=utf-8

from .conv2d import Conv2D
class MaxPool2D(Conv2D):
    def __init__(self,k_size,stride,padding):
        super(MaxPool2D,self).__init__(0,0,k_size,stride,padding)

        # useless
        del self.channels_in,self.channels_out,self.weights,self.bias
        
    def forward(self,x):
        self.last_x=x
        b,c_in,h,w=x.shape
        c_out=self.channels_out
        k_size=self.kernel_size
        
        # 1. change to matrix mul
        # [b,c_in*k_size_h*k_size_w,num_patch], num_patch=h_out*w_out
        patches,h_out,w_out=self._img2col(x)
        num_patch=h_out*w_out

        # 2. max pool 
        

    def backward(self,dx):
        pass