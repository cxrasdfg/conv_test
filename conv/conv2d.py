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
        self.bias[:]=0 # set to zero.... 

        # patches of the proceeding layer's output
        self.last_patches=None
        # output of the proceeding layer...
        self.last_x=None

    def __call__(self,x):
        return self.forward(x)

    def forward(self,x):
        r"""Forward function
        Args:
            x (th.tensor[float32]): [b,c_in,h,w]
        Return:
            x (th.tensor[float32]): [b,c_out,h_out,w_out]
        """
        self.last_x=x
        b,c_in,h,w=x.shape
        c_out=self.channels_out
        k_size=self.kernel_size
        # 1. change to matrix mul
        # [b,c_in*k_size_h*k_size_w,num_patch], num_patch=h_out*w_out
        patches,h_out,w_out=self._img2col(x)
        num_patch=h_out*w_out
        
        # 2. w @ patches + bias
        weights=self.weights.view(-1,c_in*k_size[0]*k_size[1]) # [c_out,c_in*k_size[0]*k_size[1]]
        bias=self.bias[None,:,None].expand(b,c_out,num_patch) # [b,c_out,num_patch]
        x=weights @ patches + bias # [b,c_out,num_patch]

        # 3. reshape
        x=x.view(b,c_out,h_out,w_out)

        # 4. store the patch
        self.last_patches=patches

        return x

    def backward_and_update(self,dx,lr):
        r"""Apply backward and update parameters by gradient descend
        Args:
            dx (th.tensor[float32]): dx in the succeed layer, 
        shape: [b,c_out,h_out,w_out]
            lr (float): learning rate
        Return:
            dx (th.tensor[float32]): dx in the this layer,
        shape: [b,c_out,h_in,w_in] 
        """
        k_size=self.kernel_size
        b,c_out,h_out,w_out=dx.shape
        
        # [b,c_out,h_out*w_out]
        dx=dx.view(b,c_out,-1)
        c_in=self.channels_in
        # [c_out,c_in*k_size[0]*k_size[1]]
        weights=self.weights.view(-1,c_in*k_size[0]*k_size[1]) 

        # 1. calculate dpatches
        # w @ patch + bias = x ==> dpatches=w.T @ dx
        dpatches=weights.t() @ dx  # [b,c_in*k_size[0]*k_size[1],h_out*w_out]
        dpatches=self._col2img(dpatches,self.last_x.shape) # [b,c,h,w]
        
        # 2. calculate dw
        # dw=dx @ patch.T
        # [b,c_out,c_in*k_size_h*k_size_w]
        dweight=dx @ self.last_patches.permute(0,2,1).contiguous() 
        
        # 3. calculate dbias
        # dbias=dx
        dbias=dx # [b,c_out,h_out*w_out]

        # 4. update dx, weight and bias
        dx=dpatches # [b,c,h,w]
        
        # [c_out,c_in,k_size[0],k_size[1]]
        self.weights-=lr * dweight.view(b,c_out,c_in,k_size[0],k_size[1]).sum(dim=0)

        self.bias-=lr* dbias.sum(dim=0).sum(dim=1) # [c_out]

        return dx

    def get_indices(self,h,w,k_size,stride,padding):
        r""" Get the indices for the patches 
        Args:
            h (int): height
            w (int): width
            k_size (tuple):
            stride (tuple):
            padding (tuple):
        Return:
            steps_at_i (torch.tensor[long]):
            steps_at_j (torch.tensor[long]):
            index_at_i (torch.tensor[long]): [steps_at_i,steps_at_j,k_size[0]*k_size[1]]
            index_at_j (torch.tensor[long]): [steps_at_i,steps_at_j,k_size[0]*k_size[1]]
        """
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
        idx_at_i=idx_at_i+offset_at_i
        idx_at_j=idx_at_j+offset_at_j

        # reshape
        # [steps_at_i,steps_at_j,k_size[0]*k_size[1]]
        idx_at_i=idx_at_i.view(steps_at_i,steps_at_j,-1)
        # [steps_at_i,steps_at_j,k_size[0]*k_size[1]]
        idx_at_j=idx_at_j.view(steps_at_i,steps_at_j,-1)

        return steps_at_i,steps_at_j,idx_at_i,idx_at_j

    def _pad(self,x,k_size,stride,padding):
        b,c,h,w=x.shape
        if padding[0] >0 or padding[1] >0 :
            padded_h=h+2*padding[0]
            padded_w=w+2*padding[1]
            padded_x=th.zeros(b,c,padded_h,padded_w).type_as(x)
            padded_x[:,:,padding[0]:-padding[0],padding[1]:-padding[1]]=x
        else:
            padded_x=x.clone()
        
        return padded_x

    def _img2col(self,x):
        r"""Change imgaes patches to a matrix
        Args:
            x (th.tensor[float32]): [b,c,h,w]
        Return:
            x (th.tensor[float32]): [b,c*k_size_h*k_size_w,num_patch]
            steps_at_i (int): height of the generated feature map
            steps_at_j (int): width of the generated feature map
        """
        stride=self.stride
        padding=self.padding
        k_size=self.kernel_size
        
        b,c,h,w=x.shape
        
        assert (h+2*padding[0]-k_size[0]) % stride[0] == 0
        assert (w+2*padding[1]-k_size[1]) % stride[1] == 0
        
        # 1. pad zero 
        padded_x=self._pad(x,k_size,stride,padding)

        # 2. get indices on padded_x for patches
        steps_at_i,\
            steps_at_j,\
            idx_at_i,idx_at_j=\
            self.get_indices(h,w,k_size,stride,padding)

        # 3. indexing
        # [b,c,steps_at_i,steps_at_j,k_size[0]*k_size[1]]
        patches=padded_x[...,idx_at_i,idx_at_j] 
        
        # 4. permute and reshape
        # [b,c*k_size[0]*k_size[1],steps_at_i*steps_at_j]
        patches=patches.permute(0,1,4,2,3).contiguous()\
            .view(b,c*k_size[0]*k_size[1],steps_at_i*steps_at_j)
        
        return patches,steps_at_i,steps_at_j

    def _col2img(self,patches,feat_shape):
        r"""Convert the patches to imgs by cumulative fasion
        Args:
            patches (th.tensor[float32]): [b,c*k_size[0]*k_size[1],steps_at_i*steps_at_j]
            feat_shape (tuple): (b,c,h,w)
        Return:
            x (th.tensor[float32]): [b,c,h,w]
        """
        stride=self.stride
        padding=self.padding
        k_size=self.kernel_size
        b,c,h,w=feat_shape

        # 1. allocate the memory
        padded_h=h+2*padding[0]
        padded_w=w+2*padding[1]
        padded_x=th.zeros(b,c,padded_h,padded_w).type_as(patches) # [b,c,ph,pw]

        # 2. get indices on padded_x for patches
        steps_at_i,\
            steps_at_j,\
            idx_at_i,idx_at_j=\
            self.get_indices(h,w,k_size,stride,padding)

        # 3. for `index_sum`
        padded_x=padded_x.permute(2,3,0,1).contiguous() # [ph,pw,b,c]
        padded_x=padded_x.view(-1,b,c) # [ph*pw,b,c]
    
        indices=idx_at_i*padded_w+idx_at_j # [steps_at_i,steps_at_j,k_size[0]*k_size[1]]
        indices=indices.view(-1).type_as(padded_x).long() # [steps_at_i*steps_at_j*k_size[0]*k_size[1]]
        
        # [b,c,k_size[0]*k_size[1],steps_at_i,steps_at_j]
        patches=patches.view(b,c,k_size[0]*k_size[1],steps_at_i,steps_at_j)
        # [steps_at_i,steps_at_j,k_size[0]*k_size[1],b,c]
        patches=patches.permute(3,4,2,0,1).contiguous()
        # [steps_at_i*steps_at_j*k_size[0]*k_size[1],b,c]
        patches=patches.view(-1,b,c)
        
        # 4. cumulative add...
        # NOTE: different from `padded_x[indices]+=patches`
        padded_x.index_add_(0,indices,patches)

        # 5. convert to original shape
        # [ph,pw,b,c]
        padded_x=padded_x.view(padded_h,padded_w,b,c)
        # [b,c,ph,pw]
        padded_x=padded_x.permute(2,3,0,1).contiguous()

        # 6. clip
        if padding[0] != 0:
            padded_x =padded_x[...,padding[0]:-padding[0],:] # [b,c,h,w]
        if padding[1] !=0:
            padded_x =padded_x[...,padding[1]:-padding[1]] # [b,c,h,w]
        
        return padded_x

    def cuda(self,did):
        if hasattr(self,'weights'):
            self.weights = self.weights.cuda(did)
        if hasattr(self,'bias'):
            self.bias=self.bias.cuda(did)
    
    def cpu(self):
        if hasattr(self,'weights'):
            self.weights=self.weights.cpu()
        if hasattr(self,'bias'):
            self.bias=self.bias.cpu()

def main():
    x=th.arange(2*3*3*6).view(2,3,3,6).float()
    x=th.randn(2,3,3,6).float()    
    m=Conv2D(3,5,(3,1),(2,1),(1,1))
    m2=th.nn.Conv2d(3,5,(3,1),(2,1),padding=(1,1))
    m2.weight.data.copy_(m.weights)
    m2.bias.data.copy_(m.bias)
    patches=m(x)
    patches2=m2(x)
    print(x)
    print(patches)
    print('error checking counter:',(patches!=patches2).sum() )
    dx=m.backward_and_update(patches.clone(),.1)
    print(dx)
    # cum_add=m.backward(x,patches)
    # print(cum_add)

if __name__ == '__main__':
    main()