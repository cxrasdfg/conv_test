# coding=utf-8

import torch as th
class Linear:
    def __init__(self,feat_in,feat_out):
        self.in_=feat_in
        self.out_=feat_out
        
        self.weights=th.rand([feat_out,feat_in])*2-1
        self.bias=th.randn([feat_out])
        self.bias[:]=0

        self.last_x=None

    def __call__(self,x):
        r"""
        Args:
            x (th.tensor[float32]): [b,in_]
        Return:
            x (th.tensor[float32]): [b,out_]
        """

        return self.forward(x)
    
    def forward(self,x):
        r"""
        Args:
            x (th.tensor[float32]): [b,in_]
        Return:
            x (th.tensor[float32]): [b,out_]
        """
        self.last_x=x
        x=x.t() # [in_,b]

        # [out_, b]
        x=self.weights @ x + self.bias[:,None]
        
        # [b,out_]
        x=x.t()
        return x
    
    def backward_and_update(self,dx,lr):
        r"""Backward and optimize using gradient descend
        Args:
            dx (th.tensor[float32]): [b,out_]
        Return:
            dx (th.tensor[float32]): [b,in_]
        """
        # 1. new dx
        # w @ x + bias= x ==> dx = w.T @ dx
        # [in_,b]
        new_dx=self.weights.t() @ dx.t()
        
        # 2. dweights
        # [out_,in_]
        dweights=dx.t() @ self.last_x

        # 3. dbias
        # [b,out_]
        dbias=dx
        
        # 4. update
        dx=new_dx.t() # [b,in_]
        
        self.weights-=lr*dweights
        # [out_]
        self.bias-=lr*dbias.sum(dim=0)

        return dx

    def cuda(self,did):
        self.weights = self.weights.cuda(did)
        self.bias=self.bias.cuda(did)
    
    def cpu(self):
        self.weights=self.weights.cpu()
        self.bias=self.bias.cpu()
        
def main():
    x=th.randn(2,3)
    m=Linear(3,5)
    m2=th.nn.Linear(3,5)
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

        