# coding=utf-8

class Relu:
    def __init__(self):
        self.last_x=None
        self.last_mask=None
    def __call__(self,x):
        r"""
        Args:
            x (tensor[float32]): [b,c,h,w] or [b,in_]
        Return:
            x (tensor[float32]): [b,c,h,w] or [b,in_]
        """
        return self.forward(x)
    
    def forward(self,x):
        r"""
        Args:
            x (tensor[float32]): [b,c,h,w] or [b,in_]
        Return:
            x (tensor[float32]): [b,c,h,w] or [b,in_]
        """
        self.last_x=x
        
        mask=(x>0)
        
        x[1-mask]=0

        self.last_mask=mask

        return x
    
    def backward_and_update(self,dx,lr):
        r"""
        Args:
            dx (tensor[float32]): [b,c,h,w] or [b,in_]
        Return:
            dx (tensor[float32]): [b,c,h,w] or [b,in_]
        """
        
        dx[1-self.last_mask]=0

        return dx