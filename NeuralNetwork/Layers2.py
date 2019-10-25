import numpy as np
import pyximport 
pyximport.install()
from im2col_cython import col2im_cython, im2col_cython
from im2col_cython import col2im_6d_cython
from Layers import *


class conv(object):
    def __init__(self,filters=16, filter_size=3, channels=3, width=32, height=32 ,stride=1,pad=1, std=5e-2):
        self.w = np.random.randn(filters, channels, filter_size, filter_size) * std
        self.b = np.zeros(filters)
        self.wupdate = adam(self.w)
        self.bupdate = adam(self.b)
        self.stride=stride
        self.pad=pad
        
    def layername(self):
        return "conv"
    
    
    def forward(self,x):
       # print("Shape of x is ", x.shape)
        #print("Shape of w is ", self.w.shape)
        #print("Shape of b is ", self.b.shape)

        w= self.w
        b= self.b
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = self.stride,self.pad
        num_filters, _, filter_height, filter_width = w.shape

        # Create output
        out_height = (H + 2 * pad - filter_height) / stride + 1
        out_width = (W + 2 * pad - filter_width) / stride + 1
        out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)
    
    
        # x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
        x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
       
        res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)
        

        out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
        out = out.transpose(3, 0, 1, 2)
        
        self.x=x
        self.w=w
        self.b=b
        self.x_cols =x_cols
                
       # print("Shape of out is ", out.shape)


        
        return out

   
  
    def backward(self,dout):

        x=self.x
        w=self.w
        b=self.b
        x_cols =self.x_cols
        stride, pad = self.stride,self.pad

        db = np.sum(dout, axis=(0, 2, 3))

        num_filters, _, filter_height, filter_width = w.shape
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
        dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)
        dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
        dx = col2im_cython(dx_cols, x.shape[0], x.shape[1], x.shape[2], x.shape[3],
                     filter_height, filter_width, pad, stride)
        self.w = self.wupdate.update(self.w, dw)
        self.b = self.bupdate.update(self.b, db)

        return dx
    
    def getparams(self):
        return self.w,self.b
    
    def setparams(self, w,b):
        self.w=w
        self.b=b


        
class maxpool(object):
    def forward(self,x):
        N, C, H, W = x.shape
        pool_height, pool_width = 2, 2
        stride = 2
    
        N, C, H, W = x.shape
        self.x_reshaped = x.reshape(N, C, H / pool_height, pool_height,
                               W / pool_width, pool_width)
        out = self.x_reshaped.max(axis=3).max(axis=4)
        
        self.x = x
        self.out = out
     
        return out


    def backward(self,dout):
        x_reshaped = self.x_reshaped
        x = self.x
        out = self.out
        dx_reshaped = np.zeros_like(x_reshaped)
        out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
        mask = (x_reshaped == out_newaxis)
        dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
        dx_reshaped[mask] = dout_broadcast[mask]
        dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
        dx = dx_reshaped.reshape(x.shape)

        return dx

       
    def layername(self):
        return "pool"  
    
class flatten(object):
    def __init__(self):
        a=0
    def forward(self,x):
        self.inpshape=x.shape
        return x.reshape(x.shape[0],-1)
    def backward(self, dx):
        return dx.reshape(self.inpshape)
    def layername(self):
        return "flatten"    
