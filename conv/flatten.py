# coding=utf-8

import torch as th
class Flatten:
    def __init__(self):
        self.last_x=None

    def __call__(self,x):
        r"""
        Args:
            x (th.tensor[float32]): [b,c,h,w]
        Return:
            x (th.tensor[float32]): [b,c*h*w]
        """
        return self.forward(x)

    def forward(self,x):
        r"""
        Args:
            x (th.tensor[float32]): [b,c,h,w]
        Return:
            x (th.tensor[float32]): [b,c*h*w]
        """
        b,c,w,h=x.shape
        self.last_x=x
        x=x.view(b,-1)

        return x
    
    def backward_and_update(self,dx,lr):
        r"""
        Args:
            dx (th.tensor[float32]): [b,c*h*w]
        Return:
            dx (th.tensor[float32]): [b,c,h,w]
        """

        return dx.view(self.last_x.shape)

def main():
    x=th.arange(2*3*3*6).view(2,3,3,6)
    m=Flatten()
    
    patches=m(x)
    print(x)
    print(patches)
    dx=m.backward_and_update(patches.clone(),.1)
    print(dx)

if __name__ == '__main__':
    main()