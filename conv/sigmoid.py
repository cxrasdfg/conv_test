# coding=utf-8

class Sigmoid:
    def __init__(self):
        self.last_x=None
        
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
        
        x=x.sigmoid()
        self.last_x=x
        
        return x
    
    def backward_and_update(self,dx,lr):
        r"""
        Args:
            dx (tensor[float32]): [b,c,h,w] or [b,in_]
        Return:
            dx (tensor[float32]): [b,c,h,w] or [b,in_]
        """
        
        dx=self.last_x*(1-self.last_x)*dx

        return dx