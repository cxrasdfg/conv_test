# coding=utf-8
import torch as th 
rand_seed=1234
th.manual_seed(rand_seed)
th.cuda.manual_seed(rand_seed)

from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from conv import Conv2D,MaxPool2D,Flatten,Linear,MSE,Relu,LeakyRelu,Sigmoid

IS_CUDA=True
DID=0

def create_model():
    model=[]
    # model.append(Conv2D(1,128,(3,3),(1,1),(0,0))) # [b,128,26,26]
    # model.append(Relu())
    # model.append(Conv2D(128,64,(3,3),(1,1),(0,0))) # [b,64,24,24]
    # model.append(Relu())
    # model.append(MaxPool2D((2,2),(2,2),(0,0))) # [b,64,12,12]
    # model.append(Conv2D(64,128,(3,3),(1,1),(0,0))) # [b,128,10,10]
    # model.append(Relu())

    # model.append(Conv2D(128,128,(3,3),(1,1),(0,0))) # [b,128,8,8]
    # model.append(MaxPool2D((2,2),(2,2),(0,0))) # [b,128,4,4]
    # model.append(Relu())
    # model.append(Flatten()) # [b,128*4*4]
    model.append(Linear(28*28,256))
    model.append(Sigmoid())
    # model.append(Relu())
    model.append(Linear(256,256)) # [b,512]
    model.append(Sigmoid())
    # model.append(Relu())
    model.append(Linear(256,256)) # [b,512]
    model.append(Sigmoid())
    # model.append(Relu())
    model.append(Linear(256,256)) # [b,512]
    model.append(Sigmoid())
    # model.append(Relu())
    model.append(Linear(256,512)) # [b,512]
    model.append(Sigmoid())
    # model.append(Relu())
    model.append(Linear(512,10)) # [b,10]
    model.append(Sigmoid())
    return model


def create_model_conv():
    model=[]
    model.append(Conv2D(1,10,(5,5),(1,1),(0,0))) # [b,32,26,26]
    # model.append(Sigmoid())
    model.append(Relu())
    # model.append(LeakyRelu())
    model.append(MaxPool2D((2,2),(2,2),(0,0))) # [b,64,12,12]
    model.append(Conv2D(10,20,(5,5),(1,1),(0,0))) # [b,64,24,24]
    # model.append(Sigmoid())
    model.append(Relu())
    # model.append(LeakyRelu())
    model.append(MaxPool2D((2,2),(2,2),(0,0))) # [b,64,12,12]
    model.append(Flatten()) # [b,64*12*12]
    model.append(Linear(320,50))
    # model.append(LeakyRelu())
    model.append(Relu())
    # model.append(Sigmoid())
    model.append(Linear(50,10)) # [b,10]
    model.append(Sigmoid())
    return model


def forward_model(m,x):
    for i,layer in enumerate(m):
        # if i==14:
        #     print(i)
        x=layer(x)
    return x


def cuda(model,did):
    for i in range(len(model)):
        if hasattr(model[i],'cuda'):
            model[i].cuda(DID)
        

def backward_and_optimize(m,dx,lr):
    for layer in m[::-1]:
        dx=layer.backward_and_update(dx,lr)

def preprocess(x,gt,is_conv=True):
    r"""
    Args:
        x (th.tensor[float32]): [b,28,28]
        gt (th.tensor[int]): [b]
    Return:
        x (th.tensor[float32]): [b,?]
    """
    b=x.shape[0]
    x=x*2-1
    if not is_conv:
        x=x.view(b,-1)
    gt_=th.zeros(b,10)
    gt_[th.arange(b).long(),gt]=1

    if IS_CUDA:
        x=x.cuda(DID)
        gt_=gt_.cuda(DID)
    return x,gt_

def eval(model,data_loader):
    counter=0
    tp=0.0
    for x,gt in tqdm(data_loader):
        b=x.shape[0]
        x,gt=preprocess(x,gt)
        res=forward_model(model,x) # [b,10]
        counter+=b
        tp+=float((gt.argmax(dim=1)==res.argmax(dim=1)).sum()) 
    
    return tp/counter 

def main():
    
    print('handy convolutional test...')
    lr=1e-3
    batch_size=256
    num_workers=4
    epoches=20
   

    train_data_set=MNIST('./data/',train=True,transform=ToTensor())
    test_data_set=MNIST('./data/',train=False,transform=ToTensor())

    train_data_loader=DataLoader(
        train_data_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )

    test_data_loader=DataLoader(
        test_data_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    
    model=create_model_conv()
    if IS_CUDA:
        cuda(model,DID)

    loss_func=MSE()
    epoch=0
    iteration=0
    while epoch<epoches:
        for x,gt in tqdm(train_data_loader):
            # junk...
            b=x.shape[0]
            
            # if epoch==10:
            #     print(iteration)

            x,gt=preprocess(x,gt)
            res=forward_model(model,x) # [b,10]
            loss=loss_func(res,gt)
            dx=loss_func.backward_and_update(None,None)
            backward_and_optimize(model,dx,lr)
            
            acc=((gt.argmax(dim=1)==res.argmax(dim=1)).sum() ).float()/b
            tqdm.write("Epoch:%d, Iteration:%d, Loss:%.3f, Acc:%.3f"%(epoch,iteration,loss,acc))
            iteration+=1

            # time.sleep(1)
        
        print('acc on test:%.5f' % eval(model,test_data_loader))
        epoch+=1
        


if __name__ == '__main__':
    main()
