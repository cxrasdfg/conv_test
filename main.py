# coding=utf-8
import torch as th 
rand_seed=1234
th.manual_seed(rand_seed)
th.cuda.manual_seed(rand_seed)

from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm

from conv import Conv2D,MaxPool2D,Flatten,Linear,MSE,Relu,Sigmoid

def create_model():
    model=[]
    model.append(Conv2D(1,128,(3,3),(1,1),(0,0))) # [b,128,26,26]
    model.append(Relu())
    model.append(Conv2D(128,64,(3,3),(1,1),(0,0))) # [b,64,24,24]
    model.append(Relu())
    model.append(MaxPool2D((2,2),(2,2),(0,0))) # [b,64,12,12]
    model.append(Conv2D(64,128,(3,3),(1,1),(0,0))) # [b,128,10,10]
    model.append(Relu())

    model.append(Conv2D(128,128,(3,3),(1,1),(0,0))) # [b,128,8,8]
    model.append(MaxPool2D((2,2),(2,2),(0,0))) # [b,128,4,4]
    model.append(Relu())
    model.append(Flatten()) # [b,128*4*4]
    model.append(Linear(128*4*4,512)) # [b,512]
    model.append(Relu())
    model.append(Linear(512,10)) # [b,10]
    model.append(Sigmoid())
    return model

def forward_model(m,x):
    for i,layer in enumerate(m):
        if i==14:
            print(i)
        x=layer(x)
    return x

def backward_and_optimize(m,dx,lr):
    for layer in m[::-1]:
        dx=layer.backward_and_update(dx,lr)

def main():
    
    print('handy convolutional test...')
    lr=.5
    batch_size=32
    num_workers=0
    epoches=10

    train_data_set=MNIST('./data/',train=True,transform=ToTensor())
    test_data_set=MNIST('./data/',train=False,transform=ToTensor())

    train_data_loader=DataLoader(
        train_data_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )
    
    model=create_model()
    loss_func=MSE()
    epoch=0
    iteration=0
    while epoch<epoches:
        for x,gt in tqdm(train_data_loader):
            # junk...
            gt_=th.zeros(batch_size,10)
            gt_[th.arange(batch_size).long(),gt]=1
            gt=gt_
            
            x=x*2-1
            res=forward_model(model,x) # [b,10]
            loss=loss_func(res,gt)
            dx=loss_func.backward_and_update(None,None)
            backward_and_optimize(model,dx,lr)
            tqdm.write("Epoch:%d, Iteration:%d, Loss:%.3f"%(epoch,iteration,loss))
            iteration+=1
        
        epoch+=1
        

    models=[]

if __name__ == '__main__':
    main()