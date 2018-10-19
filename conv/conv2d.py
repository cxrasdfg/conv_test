# coding=utf-8

import torch as th 

class Conv2D:
    def __init__(self,c_in,c_out,k_size,stride,padding):
        r""" Constructor
        Args:
            c_in (int): input channels
            c_out (int): output channels
            k_size (tuple): kernel size, (height,width)
            stride (tuple): stride, (h,w)
            padding (tuple): padding number
        """
        assert isinstance(c_in,int)
        assert isinstance(c_out,int)
        assert isinstance(k_size,tuple)
        assert isinstance(stride,tuple)
        assert isinstance(padding,tuple)

        self.channels_in=c_in
        self.channels_out=c_out
        self.kernel_size=k_size
        self.stride=stride
        self.padding=padding

        self.weights=th.randn([c_out,c_in,k_size[0],k_size[1]])
        self.bias=th.randn([c_out]) 

    def __call__(self,x):
        return self.forward(x)

    def forward(self,x):
        r"""Forward function
        Args:
            x (th.tensor[float32]): [b,c_in,h,w]
        Return:
            x (th.tensor[float32]): [b,c_out,h_out,w_out]
        """
        # [b,pach_len,patch_num]
        patches=self._img2col(x)

        return patches

    def backward(self,dx):
        r"""Backward function
        Args:
            dx (th.tensor[float32]): [b,c_out,h_out,w_out]
        Return:
            dx (th.tensor[float32]): [b,c_out,h_out,w_out] 
        """
    
    def _img2col(self,x):
        r"""Change imgaes patches to a matrix
        Args:
            x (th.tensor[float32]): [b,c,h,w]
        Return:
            x (th.tensor[float32]): [b,patches,c*kernel_h*kernek_w]  
        """
        stride=self.stride
        padding=self.padding
        k_size=self.kernel_size
        
        b,c,h,w=x.shape
        
        assert (h+2*padding[0]-k_size[0]) % stride[0] ==0
        assert (w+2*padding[1]-k_size[1]) % stride[1] ==0
        if padding[0] >0 or padding[1] >0 :
            padded_h=h+2*padding[0]
            padded_w=w+2*padding[1]
            padded_x=th.zeros(b,c,padded_h,padded_w).type_as(x)
            padded_x[:,:,padding[0]:-padding[0],padding[1]:-padding[1]]=x
        else:
            padded_x=x

        # 1. counter in x_axis and y_axis, the output feature map size
        steps_at_i=(h+2*padding[0]-k_size[0])// stride[0]+1
        steps_at_j=(w+2*padding[1]-k_size[1])// stride[1]+1
        
        # 2. prepare the offset
        # [steps_at_i,steps_at_j,k_size[0],k_size[1]]
        offset_at_i=(th.arange(steps_at_i)*stride[0])[:,None]\
            .expand(-1,steps_at_j)[...,None,None]\
            .expand(-1,-1,k_size[0],k_size[1]).long()
        
        # [steps_at_i,steps_at_j,k_size[0],k_size[1]]
        offset_at_j=(th.arange(steps_at_j)*stride[1])[None,:,None,None]\
            .expand_as(offset_at_i).long()
        
        # 3. prepare the index in axis x and axis y
        # [k_size[0],k_size[1]]
        idx_at_i=th.arange(k_size[0])[:,None].expand(-1,k_size[1])
        # [k_size[1],k_size[1]]
        idx_at_j=th.arange(k_size[1])[None].expand_as(idx_at_i)
        
        # [steps_at_i,steps_at_j,k_size[0],k_size[1]]
        idx_at_i=idx_at_i[None,None].expand(steps_at_i,steps_at_j,-1,-1).long()
        # [steps_at_i,steps_at_j,k_size[0],k_size[1]]
        idx_at_j=idx_at_j[None,None].expand(steps_at_i,steps_at_j,-1,-1).long()
        
        # plus the offset
        idx_at_i+=offset_at_i
        idx_at_j+=offset_at_j

        # reshape
        # [steps_at_i,steps_at_j,k_size[0]*k_size[1]]
        idx_at_i=idx_at_i.view(steps_at_i,steps_at_j,-1) 
        # [steps_at_i,steps_at_j,k_size[0]*k_size[1]]
        idx_at_j=idx_at_j.view(steps_at_i,steps_at_j,-1)

        # 4. indexing
        # [b,c,steps_at_i,steps_at_j,k_size[0]*k_size[1]]
        patches=padded_x[...,idx_at_i,idx_at_j] 
        
        # 5. permute and reshape
        # [b,c*k_size[0]*k_size[1],steps_at_i*steps_at_j]
        patches=patches.permute(0,1,4,2,3).contiguous()\
            .view(b,c*k_size[0]*k_size[1],steps_at_i*steps_at_j)
        
        
        return patches

    def _col2img(self,x):
        r"""Convert the patches to imgs by cumulative fasion
        """


def main():
    x=th.arange(2*3*3*6).view(2,3,3,6)
    patches=Conv2D(3,5,(3,1),(2,1),(1,1))(x)
    print(x)
    print(patches)

if __name__ == '__main__':
    main()