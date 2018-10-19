# coding=utf-8

import torch as th

class MSE:
    def __init__(self):
        self.last_x=None
        self.last_gt=None
    
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
        b,in_=x.shape
        self.last_x=x
        self.last_gt=gt
        loss=(x-gt)**2
        loss=loss.sum()/2/b

        return loss
    
    def backward_and_update(self,dx,lr):
        r"""
        Args:
            dx (tensor): no eggs use
            lr (float): no eggs use
        Return:
            dx (tensor[float32]): [b,in_]
        """
        dx=self.last_x-self.last_gt

        return dx

def main():
    x=th.arange(2*3).view(2,3)
    y=th.randn(2,3)
    m=MSE()
    patches=m(x,y)
    print(x)
    print(patches)
    dx=m.backward_and_update(patches.clone(),.1)
    print(dx)
    # cum_add=m.backward(x,patches)
    # print(cum_add)

if __name__ == '__main__':
    main()        
    