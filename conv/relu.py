# coding=utf-8

class LeakyRelu:
    def __init__(self,negative_slope=1e-2):
        self.last_x=None
        self.last_mask=None
        self.negative_slope=negative_slope

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
        
        mask=(x>=0)
        
        x[1-mask]*=self.negative_slope

        self.last_mask=mask

        return x
    
    def backward_and_update(self,dx,lr):
        r"""
        Args:
            dx (tensor[float32]): [b,c,h,w] or [b,in_]
        Return:
            dx (tensor[float32]): [b,c,h,w] or [b,in_]
        """
        
        dx[1-self.last_mask]*=self.negative_slope

        return dx

class Relu(LeakyRelu):
    def __init__(self):
        super(Relu,self).__init__(0)



