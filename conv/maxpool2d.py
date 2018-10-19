# coding=utf-8

import torch as th
from .conv2d import Conv2D
class MaxPool2D(Conv2D):
    def __init__(self,k_size,stride,padding):
        super(MaxPool2D,self).__init__(0,0,k_size,stride,padding)

        # useless
        del self.channels_in,self.channels_out,self.weights,self.bias
        self.last_idx=None

    def forward(self,x):
        r"""
        Args:
            x (th.tensor[float32]): [b,c_in,h_in,w_in]
        Return:
            x (th.tensor[float32]): [b,c_in,h_out,w_out]
        """
        self.last_x=x
        b,c_in,h_in,w_in=x.shape
        k_size=self.kernel_size
        
        # 1. change to matrix mul
        # [b,c_in*k_size_h*k_size_w,num_patch], num_patch=h_out*w_out
        patches,h_out,w_out=self._img2col(x)
        num_patch=h_out*w_out

        # 2. max pool 
        # [b,c,k_size_h*k_size_w,n]
        patches=patches.view(b,c_in,k_size[0]*k_size[1],h_out*w_out)
        self.last_patches=patches

        x,idx=patches.max(dim=2) # [b,c,n]
        self.last_idx=idx
        
        # 3. reshape
        x=x.view(b,c_in,h_out,w_out)

        return x

    def backward_and_update(self,dx,lr):
        r"""
        Args:
            dx (th.tensor[float32]): [b,c_out,h_out,w_out)], c_in equals to c_out
            lr (float): learning rate, no egg uses
        Return:
            dx (th.tensor[float32]): [b,c_in,h_in,w_in]
        """
        k_size=self.kernel_size
        b,c_out,h_out,w_out=dx.shape
        # [b,c,k_size_h*k_size_w,n]
        dpatches=self.last_patches
        b,c,_,n=dpatches.shape
        dpatches[:]=0
        dpatches[
            th.arange(b)[:,None,None].expand(-1,c,n).long(),
            th.arange(c)[None,:,None].expand(b,-1,n).long(),
            self.last_idx,
            th.arange(n)[None,None].expand(b,c,-1).long()]=dx.view(b,c_out,-1)
        
        # [b,c*k_size_h*k_size_w,n]
        dpatches=dpatches.view(b,-1,n)
        
        dpatches=self._col2img(dpatches,self.last_x.shape) # [b,c,h,w]
        dx=dpatches   
        return dx

def main():
    x=th.arange(2*3*4*4).view(2,3,4,4)
    m=MaxPool2D((3,3),(3,1),(1,1))
    m2=th.nn.MaxPool2d((3,3),(3,1),padding=(1,1))
    patches=m(x)
    patches2=m2(x)
    print('error checking counter:',(patches!=patches2).sum() )

    print(x)
    print(patches)
    dx=m.backward_and_update(patches.clone(),.1)
    print(dx)

if __name__ == '__main__':
    main()