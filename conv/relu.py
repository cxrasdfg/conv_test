# coding=utf-8

class LeakyRelu:
    def __init__(self,negative_slope=1e-2):
        self.last_x=None
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
        # x[1-mask]*=self.negative_slope
        x = x * (x>=0).float()+x*(x<0).float()*self.negative_slope

        return x
    
    def backward_and_update(self,dx,lr):
        r"""
        Args:
            dx (tensor[float32]): [b,c,h,w] or [b,in_]
        Return:
            dx (tensor[float32]): [b,c,h,w] or [b,in_]
        """
        
        # dx[1-self.last_mask]*=self.negative_slope
        x=self.last_x
        dx=dx*(x>=0).float() + dx*(x<0).float()*self.negative_slope

        return dx

class Relu(LeakyRelu):
    def __init__(self):
        super(Relu,self).__init__(0)



