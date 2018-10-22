# coding=utf-8

class CrossEntropy:
    def __init__(self):
        self.last_x=None
        self.last_gt=None
        self.eps=1e-6
    
    def __call__(self,x,gt):
        r"""
        Args:
            x (tensor[float32]): [b,in_]
            gt (tensor[float32]): [b,in_]
        Return:
            loss (tensor[float32]):
        """
        return self.forward(x,gt)

    def forward(self,x,gt):
        r"""
        Args:
            x (tensor[float32]): [b,in_]
            gt (tensor[float32]): [b,in_]
        Return:
            loss (tensor[float32]):
        """
        eps=self.eps
        b,in_=x.shape
        self.last_x=x
        self.last_gt=gt
        loss=(x+eps).log()*gt.type_as(x)
        loss=-loss.sum()/b

        return loss
    
    def backward_and_update(self,dx,lr):
        r"""
        Args:
            dx (tensor): no eggs use
            lr (float): no eggs use
        Return:
            dx (tensor[float32]): [b,in_]
        """
        gt=self.last_gt.type_as(self.last_x)
        x=self.last_x
        eps=self.eps
        dx=-(gt/(x+eps))
        dx=dx/self.last_x.shape[0]
        return dx
