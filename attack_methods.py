import torch
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad
from torch.optim import Adam
from ge_adv import accuracy_list,avg_accuracy
import matplotlib.pyplot as plt
import math
import os
import pickle
from tqdm import tqdm
import torch_dct as dct
import random
from scipy.fftpack import dct, idct
loss_fn = nn.CrossEntropyLoss()
epsilon = 16/255
alpha = 1.6/255

def fgsm(model, x, y, loss_fn, epsilon=epsilon):
    x_adv = x.detach().clone() # initialize x_adv as original benign image x
    x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
    out = model(x_adv)  
    loss = loss_fn(out, y)
    loss.backward() # calculate gradient
    grad = x_adv.grad.detach()
    x_adv = x_adv + epsilon * grad.sign()
    return x_adv


def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10):
    x_adv = x
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True 
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha* grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach() 
    return x_adv

def mifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True 
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        grad = x_adv.grad.detach() 
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

# Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks (ICLR 2020)'(https://arxiv.org/abs/1908.06281)
def nifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_new = x_adv + alpha*decay*momentum
        x_new.requires_grad = True 
        loss = loss_fn(model(x_new), y)
        loss.backward() 
        grad = x_new.grad.detach() 
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)      
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def vmifgsm(model, x, y, loss_fn, alpha=alpha, num_iter=10, decay=1, N = 5 ,beta = 1.5 ,epsilon=epsilon):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    v = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()  
        x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
        out = model(x_adv)  
        loss = loss_fn(out, y)                
        loss.backward() 
        
        grad1 = x_adv.grad.detach()
        grad = decay * momentum + (grad1+v)/torch.mean(torch.abs(grad1+v), dim=(1,2,3), keepdim=True)
        momentum = grad    
        # Calculate Gradient Variance
        GV_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            neighbor_images = x_adv.detach() + torch.randn_like(x).uniform_(-epsilon*beta, epsilon*beta) #ji 的把0.15改成
            neighbor_images.requires_grad = True
            out = model(neighbor_images)
            cost = loss_fn(out, y)
            cost.backward()
            grad2 = neighbor_images.grad.detach()
            GV_grad += grad2
        # obtaining the gradient variance
        v = GV_grad / N - grad1 
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        # x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv


# AITM 2022/aaai
def get_alpha(T,t_,beta1=0.9, beta2=0.99):
    res = 0
    for t in range(T):
        res += (1-beta1**(t+1))/math.sqrt(1-beta2**(t+1))
    return epsilon/res * (1-beta1**(t_+1))/math.sqrt(1-beta2**(t_+1))
    
def aitm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10,mu1=1.5,mu2=1.9, decay=1.0, beta1=0.9, beta2=0.99,r = 1.3, adam_eps=1e-8):
    x_adv = x
    #2022 AAAI Making Adversarial Examples More Transferable and Indistinguishable
    # initialize momentum tensor and second moment tensor
    momentum = torch.zeros_like(x).detach().cuda()
    v = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True   
        out = model(x_adv)  
        loss = loss_fn(out, y)                
        loss.backward()  
        grad = x_adv.grad.detach()  
        momentum = momentum + mu1*grad
        v = v + mu2 * (grad ** 2)
        m_hat =  r*( momentum/ (torch.sqrt(v) + adam_eps))
        alpha = get_alpha(num_iter,i)
        x_adv = x_adv + alpha * m_hat.tanh()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv


#NeurIPS'2023 PGN  
def pgn(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=1.5,aa=0.5,N=5):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True) 
            out = model(x_near)  
            loss = loss_fn(out, y)                
            loss.backward() 
            g1 = x_near.grad.detach() 
            x_star = x_near.detach() + alpha * (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            avg_grad += (1-aa)*g1 + aa*g2
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        #x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv

def get_cosine_similarity(cur_grad, sam_grad):
    cur_grad = cur_grad.view(cur_grad.size(0), -1)
    sam_grad = sam_grad.view(sam_grad.size(0), -1)
    cos_sim = torch.sum(cur_grad * sam_grad, dim=1) / (torch.sqrt(torch.sum(cur_grad ** 2, dim=1)) * torch.sqrt(torch.sum(sam_grad ** 2, dim=1)))
    cos_sim = cos_sim.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return cos_sim

def cap(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=3.0,aa=0.2,bb=0.5,N=20):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        sum_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            x_sun = x_near.detach() - alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            on_x = x_sun.detach()
            on_x = Variable(on_x, requires_grad=True)
            out = model(on_x)
            loss = loss_fn(out, y)
            loss.backward()
            g3 = on_x.grad.detach()
            sum_grad += (1-aa-bb)*g1 + aa*g2 + bb*g3
        avg_grad = sum_grad/N
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv


def cosine_similarity(g1, g2):
    dot_product = torch.sum(g1 * g2, dim=(1, 2, 3))  # 计算点积
    norm_g1 = torch.norm(g1, p=2, dim=(1, 2, 3))  # 计算L2范数
    norm_g2 = torch.norm(g2, p=2, dim=(1, 2, 3))  # 计算L2范数
    return dot_product / (norm_g1 * norm_g2)

def gvmi_fgsm(model, x, y, loss_fn, alpha=alpha, num_iter=10, decay=1, N=20, beta=3/2, epsilon=epsilon, P = 5, S = 10):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    v = torch.zeros_like(x).detach().cuda()
    for i in range(P):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True  # need to obtain gradient of x_adv, thus set required grad
        out = model(x_adv)
        loss = loss_fn(out, y)
        loss.backward()

        grad1 = x_adv.grad.detach()
        grad = decay * momentum + (grad1 + v) / torch.mean(torch.abs(grad1 + v), dim=(1, 2, 3), keepdim=True)
        momentum = grad
        # Calculate Gradient Variance
        GV_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            neighbor_images = x_adv.detach() + torch.randn_like(x).uniform_(-epsilon * beta,
                                                                            epsilon * beta)  # ji 的把0.15改成
            neighbor_images.requires_grad = True
            out = model(neighbor_images)
            cost = loss_fn(out, y)
            cost.backward()
            grad2 = neighbor_images.grad.detach()
            GV_grad += grad2
        # obtaining the gradient variance
        v = GV_grad / N - grad1
        x_adv = x_adv + S * alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        # x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    x_adv = x
    for j in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True  # need to obtain gradient of x_adv, thus set required grad
        out = model(x_adv)
        loss = loss_fn(out, y)
        loss.backward()

        grad1 = x_adv.grad.detach()
        grad = decay * momentum + (grad1 + v) / torch.mean(torch.abs(grad1 + v), dim=(1, 2, 3), keepdim=True)
        momentum = grad
        # Calculate Gradient Variance
        GV_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            neighbor_images = x_adv.detach() + torch.randn_like(x).uniform_(-epsilon * beta,
                                                                            epsilon * beta)  # ji 的把0.15改成
            neighbor_images.requires_grad = True
            out = model(neighbor_images)
            cost = loss_fn(out, y)
            cost.backward()
            grad2 = neighbor_images.grad.detach()
            GV_grad += grad2
        # obtaining the gradient variance
        v = GV_grad / N - grad1
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        # x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv


#  2023/ICCV 论文GRA
def get_decay_indicator(M, x, cur_noise, last_noise, eta,):
    
    if isinstance(last_noise, int):
        last_noise = torch.full(cur_noise.shape, last_noise)
    else:
        last_noise = last_noise
    if torch.cuda.is_available():
        last_noise = last_noise.cuda()
    last = last_noise.sign()
    cur = cur_noise.sign()
    eq_m = (last == cur).float()
    di_m = torch.ones_like(x) - eq_m
    M = M * (eq_m + di_m * eta)
    return M    
    
def gra(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha,N=5,num_iter=10, decay=1.0, beta= 1.5,eta=0.94):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    
    # Initialize the decay indicator
    M = torch.full_like(x, 1 / eta)    
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()   
        x_adv.requires_grad = True
        out = model(x_adv)
        cost = loss_fn(out, y)
        cost.backward()
        grad1 = x_adv.grad.detach()
        GV_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            
            neighbor_images = x_adv.detach() + torch.randn_like(x).uniform_(-epsilon*beta, epsilon*beta) #ji 的把0.15改成
            neighbor_images.requires_grad = True
            out = model(neighbor_images)
            cost = loss_fn(out, y)
            cost.backward()
            grad_n = neighbor_images.grad.detach()
            GV_grad += grad_n
        grad_n = GV_grad / N    
        cossim = get_cosine_similarity(grad1,grad_n)
        c_grad = cossim*grad1 + (1-cossim)*grad_n
        last_momentum = momentum
        grad = decay*momentum +(c_grad)/torch.mean(torch.abs(c_grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        
        M = get_decay_indicator(M,x, momentum, last_momentum, eta)
        x_adv = x_adv + alpha*M*grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv



# ICCV'2023   Structure Invariant Transformation for better Adversarial Transferability  SIA
def sia(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha,num_copies=5,num_block=3, num_iter=10, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    def vertical_shift( x):
        _, _, w, _ = x.shape
        step = np.random.randint(low = 0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)
    def horizontal_shift( x):
        _, _, _, h = x.shape
        step = np.random.randint(low = 0, high=h, dtype=np.int32)
        return x.roll(step, dims=3)
    def vertical_flip( x):
        return x.flip(dims=(2,))
    def horizontal_flip( x):
        return x.flip(dims=(3,))
    def rotate180( x):
        return x.rot90(k=2, dims=(2,3))
    def scale( x):
        return torch.rand(1)[0] * x
    def resize( x):
        """
        Resize the input
        """
        _, _, w, h = x.shape
        scale_factor = 0.8
        new_h = int(h * scale_factor)+1
        new_w = int(w * scale_factor)+1
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=False).clamp(0, 1)
        return x
    
    def dct( x):
        """
        Discrete Fourier Transform
        """
        dctx = torch.fft.fftn(x, dim=(-2, -1))
        
        #dctx = dct.dct_2d(x) #torch.fft.fftn(x, dim=(-2, -1))
        _, _, w, h = dctx.shape
        low_ratio = 0.4
        low_w = int(w * low_ratio)
        low_h = int(h * low_ratio)
        # dctx[:, :, -low_w:, -low_h:] = 0
        dctx[:, :, -low_w:,:] = 0
        dctx[:, :, :, -low_h:] = 0
        dctx = dctx # * mask.reshape(1, 1, w, h)
        
        idctx = torch.fft.ifftn(x, dim=(-2, -1))
        #idctx = dct.idct_2d(dctx)
        return idctx
    def add_noise( x):
        return torch.clip(x + torch.zeros_like(x).uniform_(-16/255,16/255), 0, 1)
    def gkern( kernel_size=3, nsig=3):
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).cuda()

    def drop_out( x):
        
        return F.dropout2d(x, p=0.1, training=True) 
    def blocktransform( x, choice=-1):
        _, _, w, h = x.shape
        y_axis = [0,] + np.random.choice(list(range(1, h)), num_block-1, replace=False).tolist() + [h,]
        x_axis = [0,] + np.random.choice(list(range(1, w)), num_block-1, replace=False).tolist() + [w,]
        y_axis.sort()
        x_axis.sort()
        
        x_copy = x.clone()
        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(op), dtype=np.int32)
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = op[chosen](x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])

        return x_copy
    def transform( x):
        """
        Scale the input for BlockShuffle
        """
        return torch.cat([blocktransform(x) for _ in range(num_copies)])

    for i in range(num_iter):
        x_adv = x_adv.detach().clone()      
        x_adv.requires_grad = True 
       
        op = [resize,vertical_shift,horizontal_shift,vertical_flip,horizontal_flip,rotate180,scale,add_noise,drop_out,dct]#add_noise,drop_out
        x_other = transform(x_adv)
        out = model(x_other)    
        #loss = loss_fn(out, y)        
        loss = loss_fn(out, y.repeat(num_copies))       
        loss.backward() 
        grad = x_adv.grad.detach() 
        grad = decay * momentum +  grad / (grad.abs().sum() + 1e-8)       
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        #x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv















#Y  cvpr2024
def get_length(length):
    num_block=3
    rand = np.random.uniform(size=num_block)
    rand_norm = np.round(rand/rand.sum()*length).astype(np.int32)
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)

def shuffle_single_dim(x, dim):
    lengths = get_length(x.size(dim))
    x_strips = list(x.split(lengths, dim=dim))
    random.shuffle(x_strips)
    return x_strips


def shuffle1(x):
    dims = [2,3]
    random.shuffle(dims)
    x_strips = shuffle_single_dim(x, dims[0])
    return torch.cat([torch.cat(shuffle_single_dim(x_strip, dim=dims[1]), dim=dims[1]) for x_strip in x_strips], dim=dims[0])



def bsr(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, M=20,num_iter=10, num_block=3,decay=1.0):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        grad_m = 0
        for n in range(M):
            x_new=shuffle1(x_adv) 
            out = model(x_new)  
            loss = loss_fn(out, y)
            loss.backward()
            grad_m += x_adv.grad.detach()
        grad = grad_m / M
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)      
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def si_fgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, M=5,num_iter=10,decay=1.0):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        
        grad_m = 0
        for n in range(M):
            x_new= x_adv/(2**n) 
            x_new.requires_grad = True
            out = model(x_new)  
            loss = loss_fn(out, y)
            loss.backward()
            grad_m += x_new.grad.detach()
        grad = grad_m / M
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)      
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def DI(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    ret = padded if torch.rand(1) < diversity_prob else x
    return ret

def dicap(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=3.0,aa=0.2,bb=0.5,N=20):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        sum_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(DI(x_near))
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(DI(nes_x))
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            x_sun = x_near.detach() - alpha * g1 / torch.abs(g1).mean([1, 2, 3], keepdim=True)
            on_x = x_sun.detach()
            on_x = Variable(on_x, requires_grad=True)
            out = model(DI(on_x))
            loss = loss_fn(out, y)
            loss.backward()
            g3 = on_x.grad.detach()
            sum_grad += (1-aa-bb)*g1 + aa*g2 + bb*g3
        avg_grad = sum_grad / N
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def gkern(kernlen=5, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def TI(grad_in, kernel_size=5):
    kernel = gkern(kernel_size, 3).astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()
    #conv2d(grad2, TI_kernel(), bias=None, stride=1, padding=(2,2), groups=3)
    grad_out = F.conv2d(grad_in, gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3) #TI
    return grad_out

def ticap(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=3.0,aa=0.2,bb=0.5,N=20):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        sum_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            x_sun = x_near.detach() - alpha * g1 / torch.abs(g1).mean([1, 2, 3], keepdim=True)
            on_x = x_sun.detach()
            on_x = Variable(on_x, requires_grad=True)
            out = model(on_x)
            loss = loss_fn(out, y)
            loss.backward()
            g3 = on_x.grad.detach()
            sum_grad += (1-aa-bb)*g1 + aa*g2 + bb*g3
        avg_grad = sum_grad / N
        avg_grad = TI(avg_grad)
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def sicap(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=3.0,aa=0.2,bb=0.5,N=20):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        sum_grad = torch.zeros_like(x).detach().cuda()
        for m in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(x_near/(2**m))
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x/(2**m))
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            x_sun = x_near.detach() - alpha * g1 / torch.abs(g1).mean([1, 2, 3], keepdim=True)
            on_x = x_sun.detach()
            on_x = Variable(on_x, requires_grad=True)
            out = model(on_x/(2**m))
            loss = loss_fn(out, y)
            loss.backward()
            g3 = on_x.grad.detach()
            sum_grad += (1-aa-bb)*g1 + aa*g2 + bb*g3
        avg_grad = sum_grad / N
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def mix(x_adv):   # 跟dmi的用法一样  直接放到模型的输出哪里  model(mix(xadv))
    img_other = x_adv[torch.randperm(x_adv.shape[0])].view(x_adv.size())
    xx= x_adv + 0.2 * img_other
    return xx

def adcap(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=3.0,aa=0.2,bb=0.5,N=20):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        sum_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(mix(x_near))
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(mix(nes_x))
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            x_sun = x_near.detach() - alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            on_x = x_sun.detach()
            on_x = Variable(on_x, requires_grad=True)
            out = model(mix(on_x))
            loss = loss_fn(out, y)
            loss.backward()
            g3 = on_x.grad.detach()
            sum_grad += (1-aa-bb)*g1 + aa*g2 + bb*g3
        avg_grad = sum_grad/N
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def ssa(x,rho=0.5):
    _,_,_,image_width = x.size()
    gauss = torch.randn(x.size()[0], 3, image_width, image_width) * epsilon
    gauss = gauss.cuda()
    x_dct = dct_2d(x + gauss).cuda()
    mask = (torch.rand_like(x) * 2 * rho + 1 - rho).cuda()
    x_idct = idct_2d(x_dct * mask)
    return x_idct

def ssacap(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=3.0,aa=0.2,bb=0.5,N=20):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        sum_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(ssa(x_near))
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            x_sun = x_near.detach() - alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            on_x = x_sun.detach()
            on_x = Variable(on_x, requires_grad=True)
            out = model(on_x)
            loss = loss_fn(out, y)
            loss.backward()
            g3 = on_x.grad.detach()
            sum_grad += (1-aa-bb)*g1 + aa*g2 + bb*g3
        avg_grad = sum_grad/N
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def dim(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(DI(x_adv)), y)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def tim(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = TI(grad)
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def sim(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for m in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(x_adv/(2**m)), y)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def admix(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(mix(x_adv)), y)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.fft.fft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1).real.view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape).real


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)

def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)

def SSA(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(ssa(x_adv)), y)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv


def LAW_T(model, x, y, loss_fn):
    """
    Placeholder for L_AWT in the paper.
    You can replace this with the exact definition.
    """
    logits = model(x)
    return loss_fn(logits, y)

def awt(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=0.005,aa=0.5,N=20,lr=0.002):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()
        # ===== Line 4: θ̂s = θs + β · ∇θs L_AWT =====
        loss = LAW_T(model, x_adv, y, loss_fn)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        with torch.no_grad():
            for p, g_p in zip(model.parameters(), grads):
                p.add_(beta * g_p)

        # ===== Line 5: θs ← θs − lr · ∇θ̂s L_AWT =====
        loss_hat = LAW_T(model, x_adv, y, loss_fn)
        grads_hat = torch.autograd.grad(loss_hat, model.parameters())

        with torch.no_grad():
            for p, g_p in zip(model.parameters(), grads_hat):
                p.sub_(lr * g_p)

        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*3.0, epsilon*3.0)
            x_near = Variable(x_near, requires_grad = True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            avg_grad += (1-aa)*g1 + aa*g2
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        #x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv
from copy import deepcopy

def awt2(
    model, x, y, loss_fn,
    epsilon=epsilon, alpha=alpha,
    num_iter=10, decay=1.0,
    beta=0.005, omega=0.5,
    N=20, lr=0.002, zeta=3*epsilon
):
    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x).cuda()

    for _ in range(num_iter):

        theta_backup = deepcopy(model.state_dict())
        avg_grad = torch.zeros_like(x).cuda()

        # ===== parameter perturbation =====
        loss = LAW_T(model, x_adv, y, loss_fn)
        grads = torch.autograd.grad(
            loss,
            model.parameters(),
            create_graph=True,
            allow_unused=True
        )

        with torch.no_grad():
            for p, g in zip(model.parameters(), grads):
                if g is not None:
                    p.add_(beta * g)

        loss_hat = LAW_T(model, x_adv, y, loss_fn)
        grads_hat = torch.autograd.grad(
            loss_hat,
            model.parameters(),
            allow_unused=True
        )

        with torch.no_grad():
            for p, g in zip(model.parameters(), grads_hat):
                if g is not None:
                    p.sub_(lr * g)

        # ===== random sampling =====
        for _ in range(N):
            x_near = (x_adv + torch.randn_like(x) * zeta).detach().requires_grad_(True)
            loss = loss_fn(model(x_near), y)
            g1 = torch.autograd.grad(loss, x_near)[0]

            x_pred = x_near - alpha * g1 / (g1.abs().sum([1,2,3], keepdim=True) + 1e-12)
            x_pred = x_pred.detach().requires_grad_(True)
            loss = loss_fn(model(x_pred), y)
            g2 = torch.autograd.grad(loss, x_pred)[0]

            avg_grad += (1 - omega) * g1 + omega * g2

        grad = avg_grad / (avg_grad.abs().sum([1,2,3], keepdim=True) + 1e-12)
        grad = decay * momentum + grad
        momentum = grad

        x_adv = x_adv + alpha * torch.sign(grad)
        x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1).detach()

        model.load_state_dict(theta_backup)

    return x_adv
def HV(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=3.0,aa=0.5,N=10,beta1=0.9,beta2=0.999):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()
        m0=torch.zeros_like(x).detach().cuda()
        a0 = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()

            # mt=beta1*m0+(1-beta1)*g1
            # at=beta2*a0+(1-beta2)*g2*g2
            # m0=mt
            # a0=at
            # pt=(g1-grad)*(g1-grad)
            # eta=pt/((g1-g2)*(g1-g2)+gama*pt)
            # st=beta2*s0+(1-beta2)*eta*pt
            x_star = x_near.detach() + alpha * (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()

            mt=beta1*m0+(1-beta1)*g1
            at=beta2*a0+(1-beta2)*(g2**2)
            m0=mt
            a0=at
            # pt=(g1-grad)*(g1-grad)
            # eta=pt/((g1-g2)*(g1-g2)+gama*pt)
            # st=beta2*s0+(1-beta2)*eta*pt
            m_hat = mt / (1 - beta1 ** N)
            v_hat = at /(1 - beta2 ** N)



            avg_grad += (1-aa)*m_hat + aa*v_hat**(-0.5)

        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        #x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv

#g1取负值
def gaa(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=1.5,ro=1.6/255,la=0.2,N=5):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()

            x_star = x_near.detach() + ro * (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()

            avg_grad += g2+(1-la)*g1 + (1+la)*g2
        avg_grad = avg_grad/N #没有作用
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        #x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv

#复现2025 iccv Enhancing Adversarial Transferability by Balancing Exploration and  Exploitation with Gradient-Guided Sampling
def ggs(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, N=5,zeta=1.5):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()
        #x_near = Variable(x_near, requires_grad=True)
        for i in range(1,N+1):
            noise = torch.zeros_like(x).uniform_(-zeta * epsilon, zeta * epsilon)
            if i != 1 :
                noise = noise.abs() * g1.detach().sign()
            noise = noise.clamp(-zeta * epsilon, zeta * epsilon)

            x_near = x_adv + noise
            x_near = Variable(x_near, requires_grad=True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()

            # x_star = x_near.detach() + (-g1) / torch.abs(g1).mean([1, 2, 3], keepdim=True)
            # nes_x = x_star.detach()
            # nes_x = Variable(nes_x, requires_grad = True)
            # out = model(nes_x)
            # loss = loss_fn(out, y)
            # loss.backward()
            # g2 = nes_x.grad.detach()



            avg_grad += g1

        avg_grad = avg_grad / N
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        # x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv

def ggs1(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, N=10,zeta=1.5):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()
        #x_near = Variable(x_near, requires_grad=True)
        for i in range(1,N+1):
            noise = torch.zeros_like(x).uniform_(-zeta * epsilon, zeta * epsilon)
            if i != 1 :
                noise = noise.abs() * g1.detach().sign()
            noise = noise.clamp(-zeta * epsilon, zeta * epsilon)

            x_near = x_adv + noise
            x_near = Variable(x_near, requires_grad=True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()

            # x_star = x_near.detach() + (-g1) / torch.abs(g1).mean([1, 2, 3], keepdim=True)
            # nes_x = x_star.detach()
            # nes_x = Variable(nes_x, requires_grad = True)
            # out = model(nes_x)
            # loss = loss_fn(out, y)
            # loss.backward()
            # g2 = nes_x.grad.detach()



            avg_grad += g1

        avg_grad = avg_grad / N
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        # x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv

def ggs2(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, N=20,zeta=1.5):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()
        #x_near = Variable(x_near, requires_grad=True)
        for i in range(1,N+1):
            noise = torch.zeros_like(x).uniform_(-zeta * epsilon, zeta * epsilon)
            if i != 1 :
                noise = noise.abs() * g1.detach().sign()
            noise = noise.clamp(-zeta * epsilon, zeta * epsilon)

            x_near = x_adv + noise
            x_near = Variable(x_near, requires_grad=True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()

            # x_star = x_near.detach() + (-g1) / torch.abs(g1).mean([1, 2, 3], keepdim=True)
            # nes_x = x_star.detach()
            # nes_x = Variable(nes_x, requires_grad = True)
            # out = model(nes_x)
            # loss = loss_fn(out, y)
            # loss.backward()
            # g2 = nes_x.grad.detach()



            avg_grad += g1

        avg_grad = avg_grad / N
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        # x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv

def path_min_integrated(model, y, x_near, s, g_min):
    g_avg = torch.zeros_like(x_near).detach().cuda()
    for t in torch.linspace(0, 1, s):
        x_star = x_near.detach() + alpha *t*g_min
        nes_x = x_star.detach()
        nes_x = Variable(nes_x, requires_grad = True)
        out = model(nes_x)
        loss = loss_fn(out, y)
        loss.backward()
        g2 = nes_x.grad.detach()
        g_avg += g2   #加了权重
    g_avg = g_avg/s
    return g_avg

#复现 2026 AAAI  Prompting Adversarial Transferability via Path Flatness Attack
# S = 5的时候优于20
def pfa(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=1.5,aa=0.1,N=10, s=5):#原文N为20 yuan zeta = 3
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()


            g_min = (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)
           # g_max = (g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)


            g_inte = path_min_integrated( model, y, x_near, s, g_min)

            avg_grad += (1-aa)*g_inte + aa*g1 #  分数超级高

        avg_grad=avg_grad/N

        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        #x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv
def path_min_integrated1(model, y, x_near, s, g_min):
    g_avg = torch.zeros_like(x_near).detach().cuda()
    for t in torch.linspace(0, 1, s):
        x_star = x_near.detach() + alpha *t*g_min

        nes_x = x_star.detach()
        nes_x = Variable(nes_x, requires_grad = True)
        out = model(nes_x)
        loss = loss_fn(out, y)
        loss.backward()
        g2 = nes_x.grad.detach()
        g_avg += g2   #加了权重
        x_near = x_star
    g_avg = g_avg/s
    return g_avg


# S = 5的时候优于20
def jifen1(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=1.5,aa=0.1,N=5, s=5):#原文N为20 yuan zeta = 3
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()


            g_min = (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)
           # g_max = (g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)


            g_inte = path_min_integrated1( model, y, x_near, s, g_min)

            avg_grad += (1-aa)*g_inte + aa*g1 #  分数超级高

        avg_grad=avg_grad/N

        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        #x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv

#复现2025 TIFS
def mef(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decayin=0.9,decayout=0.5,beta=1.5, kesai=0.15, N=5):
    x_adv = x
    # initialze momentum tensor
    #momentum = torch.zeros_like(x).detach().cuda()
    grad_pre = torch.zeros_like(x).detach().cuda()
    grad_t = torch.zeros_like(x).detach().cuda()
    b, c, h, w = x_adv.shape

    grad_list = torch.zeros([N, b, c, h, w]).cuda()
    grad_pgia = torch.zeros([N, b, c, h, w]).cuda()

    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()

        # x_near = Variable(x_near, requires_grad=True)
        for k in range(N):
            x_near = x_adv + torch.rand_like(x_adv).uniform_(-epsilon * beta, epsilon * beta)
            x_min = x_near + kesai*epsilon*(grad_pgia[k])

            x_min = Variable(x_min, requires_grad=True)
            out = model(x_min)
            loss = loss_fn(out, y)
            loss.backward()
            grad_list[k] = x_min.grad.detach().clone()
            x_min.grad.zero_()




        grad = grad_list * (1/N)

        grad_pgia = ((grad / torch.mean(torch.abs(grad), (2, 3, 4), keepdim=True)) - decayin * grad_pgia)
        grad_t = grad.sum(0)
        grad_t = grad_t / torch.mean(torch.abs(grad_t), (1, 2, 3), keepdim=True)
        input_grad = grad_t + decayout * grad_pre
        grad_pre = input_grad
        input_grad = input_grad / torch.mean(torch.abs(input_grad), (1, 2, 3), keepdim=True)



        x_adv = x_adv + alpha * torch.sign(input_grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        # x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]

    return x_adv


#被抛弃的方案
def lgs7_1(model,x,y,loss_fn,epsilon=epsilon,alpha=alpha,num_iter=10,decay=1.0,beta=2.0,N=20,zeta=2.0,
         gamma=0.5,aa=0.5,K=4,lam=0.05,feature_fn=None,use_sign_bias=True,reg_norm=True):

    device = x.device
    x_adv = x.detach().clone()
    momentum = torch.zeros_like(x, device=device)

    for _ in range(num_iter):
        avg_grad = torch.zeros_like(x, device=device)
        prev_sign = None

        for i in range(1, N + 1):
            # 1) directional noise
            noise = torch.empty_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)

            if i != 1 :
                noise = noise.abs() * prev_sign

            noise = noise.clamp(-zeta * epsilon, zeta * epsilon)

            # 2) 基础采样点
            x_base = x_adv + noise
            delta_base = torch.clamp(x_base - x, min=-epsilon, max=epsilon)
            x_base = torch.clamp(x + delta_base, min=0.0, max=1.0).detach().requires_grad_(True)

            # 3) 分类损失
            out = model(x_base)
            cls_loss = loss_fn(out, y)

            # 4) min ||f1-f2|| 正则
            reg_each,noise= get_min_feature_reg1(
                model=model,
                x_adv=x_adv,
                x=x,
                epsilon=epsilon,
                zeta=zeta,
                K=K,
                feature_fn=feature_fn
            )  # [B]

            # if reg_norm:
            #     reg_each = reg_each / (reg_each.detach().mean() )

            reg = reg_each.mean()

            # x_base = x_adv + noise
            # delta_base = torch.clamp(x_base - x, min=-epsilon, max=epsilon)
            # x_base = torch.clamp(x + delta_base, min=0.0, max=1.0).detach().requires_grad_(True)
            #
            # # 3) 分类损失
            # out = model(x_base)
            # cls_loss = loss_fn(out, y)

            # 5) 联合目标
            total_loss = cls_loss - lam * reg


            # 6) 对采样点求梯度
            # g1 = torch.autograd.grad(
            #     total_loss,
            #     x_base,
            #     retain_graph=False,
            #     create_graph=False
            # )[0]


            total_loss.backward()
            g1 = x_base.grad.detach()
            prev_sign = g1.detach().sign()
            # PGN套件
            x_star = x_base.detach() + alpha * (-g1) / (torch.abs(g1).mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
            delta_star = torch.clamp(x_star - x, min=-epsilon, max=epsilon)
            x_star = torch.clamp(x + delta_star, min=0.0, max=1.0)

            nes_x = Variable(x_star.detach(), requires_grad=True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()

            avg_grad += (1 - aa) * g1 + aa * g2


        # 7) 梯度平均 + 动量
        avg_grad = avg_grad / N
        grad = avg_grad / (avg_grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
        grad = decay * momentum + grad
        momentum = grad

        # 8) 更新对抗样本
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0.0, max=1.0).detach()

    return x_adv




def sfa(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=10,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv


#复现 2025 iccv Boosting Adversarial Transferability via Residual Perturbation Attack
def respa(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, zeta=1.5, beta1=0.6,delta=0.6,N=5):
    x_adv = x
    grad = torch.zeros_like(x).detach().cuda()
    M = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*zeta, epsilon * zeta)
            x_near = Variable(x_near, requires_grad=True)
            output_v3 = model(x_near)
            loss = loss_fn(output_v3, y)
            g1 = torch.autograd.grad(loss, x_near,
                                     retain_graph=False, create_graph=False)[0]
            M1 = beta1 * M + (1 - beta1) * g1
            g11 = g1 - M1
            x_star = x_near.detach() + alpha * (-g11) / torch.abs(g11).mean([1, 2, 3], keepdim=True)

            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad=True)
            output_v3 = model(nes_x)
            loss = loss_fn(output_v3, y)
            g2 = torch.autograd.grad(loss, nes_x,
                                     retain_graph=False, create_graph=False)[0]

            avg_grad += (1 - delta) * g1 + delta * g2
        avg_grad = avg_grad / N
        M = beta1 * M + (1 - beta1) * avg_grad
        noise = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        noise = decay * grad + noise
        grad = noise

        x_adv = x_adv + alpha * torch.sign(noise)
        delta1= torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta1, min=0, max=1).detach()
    return x_adv
#复现2025TIFS  Boosting the Transferability of Adversarial  Examples Through Gradient Aggregation
def gaa(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=1.5,lam=0.2,N=5):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * (g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            avg_grad += g2 + (1-lam) * g1 + (1+lam) * g2
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()

    return x_adv

# def two_3(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.08, N=20,zeta=2):
#     device = x.device
#     x_adv = x
#
#     momentum = torch.zeros_like(x).detach().cuda()
#     for _ in range(num_iter):
#         x_adv = x_adv.detach().clone()
#         avg_grad = torch.zeros_like(x).detach().cuda()
#
#         #x_near = Variable(x_near, requires_grad=True)
#         for i in range(N):
#             if i==0:
#                 g3 = torch.zeros_like(x).detach().cuda()
#             x = Variable(x_adv, requires_grad=True)
#             out1 = model(x)
#             loss1 = loss_fn(out1, y)
#             loss1.backward()
#             g1 = x.grad.detach()
#
#             noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
#             z = x_adv  + noise.abs() * g3.detach().sign()
#             z = Variable(z, requires_grad=True)
#             out2 = model(z)
#             loss2 = loss_fn(out2, y)
#             loss2.backward()
#             g2 = z.grad.detach()
#
#             f = torch.sign(loss2 - loss1)
#
#             g3 = (1 - beta * f) * g2 + beta * f *g1
#
#
#             avg_grad += g3
#
#
#
#         grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
#         grad = decay * momentum + grad
#         momentum = grad
#         x_adv = x_adv + alpha * torch.sign(grad)
#         delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
#         x_adv = torch.clamp(x + delta, min=0, max=1).detach()
#     return x_adv
def dimsfa(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=5,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()
            x = Variable(x_adv, requires_grad=True)
            out1 = model(DI(x))
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(DI(z))
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def timsfa(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=5,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()
            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        grad = TI(grad)
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def simsfa(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=5,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()
            x = Variable(x_adv, requires_grad=True)
            out1 = model(x/(2**i))
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z/(2**i))
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def admixsfa(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=5,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()
            x = Variable(x_adv, requires_grad=True)
            out1 = model(mix(x))
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(mix(z))
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def ssasfa(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=5,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()
            x = Variable(x_adv, requires_grad=True)
            out1 = model(ssa(x))
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(ssa(z))
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv




def SFA1(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=10,zeta=0.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA2(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=10,zeta=1):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA3(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=10,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA4(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=10,zeta=2):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA5(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=10,zeta=2.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA6(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=10,zeta=3):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA7(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=10,zeta=3.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA8(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=10,zeta=4):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA9(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=10,zeta=4.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def SFA01(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=1,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA02(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=5,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA03(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=10,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA04(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=15,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA05(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=20,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA06(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=25,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA07(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=30,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA08(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=35,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA09(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=40,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA010(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=45,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA011(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=50,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv


def SFA001(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0, N=10,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA002(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.1, N=10,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA003(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.2, N=10,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA004(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.3, N=10,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA005(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.4, N=10,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA006(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.5, N=10,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA007(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.6, N=10,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA008(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.7, N=10,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA009(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.8, N=10,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA0010(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=0.9, N=10,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
def SFA0011(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=1, N=10,zeta=1.5):
    device = x.device
    x_adv = x

    momentum = torch.zeros_like(x).detach().cuda()
    for _ in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()

        #x_near = Variable(x_near, requires_grad=True)
        for i in range(N):
            if i==0:
                g3 = torch.zeros_like(x).detach().cuda()

            x = Variable(x_adv, requires_grad=True)
            out1 = model(x)
            loss1 = loss_fn(out1, y)
            loss1.backward()
            g1 = x.grad.detach()

            noise = torch.zeros_like(x_adv).uniform_(-zeta * epsilon, zeta * epsilon)
            z = x_adv  + noise.abs() * g3.detach().sign()
            z = Variable(z, requires_grad=True)
            out2 = model(z)
            loss2 = loss_fn(out2, y)
            loss2.backward()
            g2 = z.grad.detach()

            f = torch.sign(loss2 - loss1)

            g3 = (1 - beta * f) * g2 + beta * f *g1


            avg_grad += g3
        avg_grad = avg_grad/N



        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv


