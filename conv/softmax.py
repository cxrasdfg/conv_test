# coding=utf-8
import torch as th

class Softmax:
    def __init__(self):
        self.last_x=None
        
    def __call__(self,x):
        r"""
        Args:
            x (tensor[float32]): [b,in_]
        Return:
            x (tensor[float32]): [b,in_]
        """
        return self.forward(x)
    
    def forward(self,x):
        r"""
        Args:
            x (tensor[float32]): [b,in_]
        Return:
            x (tensor[float32]): [b,in_]
        """
        assert x.dim()==2
        # x=x.softmax(dim=1)
        x=th.nn.functional.softmax(x,dim=1)
        self.last_x=x
        return x
    
    def backward_and_update(self,dx,lr):
        r"""
        Args:
            dx (tensor[float32]): [b,in_]
        Return:
            dx (tensor[float32]): [b,in_]
        """
        y=self.last_x
        # dx=self.last_x*(1-self.last_x)*dx
        b,in_=y.shape
        _diag=th.zeros(b,in_,in_).type_as(dx)
        _diag[:,th.arange(in_).long(),th.arange(in_).long()]=y
        dx=(_diag-y[...,None].expand(-1,-1,in_)\
            *y[:,None,:].expand(-1,in_,-1))\
            *dx[...,None].expand(-1,-1,in_)
        dx=dx.sum(dim=1) # [b,in_]
        return dx