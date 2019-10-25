
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class layer(object):
    def __init__(self, inpnum, neuronsnum):
        self.w = np.random.randn(inpnum, neuronsnum)/np.sqrt(inpnum/2.0)
        self.b = np.zeros(neuronsnum)
        self.wupdate = adam(self.w)
        self.bupdate = adam(self.b)
        self.x = None
    
    def layername(self):
        return "layer"    
    
    def forward(self,x):
        out = None
        N = x.shape[0]
        x_temp= x.reshape((N, -1))
        out= np.dot(x_temp, self.w) + self.b

        self.x = x
       # print (self.w)
    
        return out
    
    def backward(self,dout):
        x = self.x
        dx,dw,db = None,None,None
        x_shape = x.shape
        N = x.shape[0]
        x= x.reshape((N, -1))
        dx = np.dot(dout, self.w.T).reshape(x_shape)
        
        dw = np.dot(x.T, dout)
        db = np.sum(dout, axis=0)
        
        self.w = self.wupdate.update(self.w, dw)
        self.b = self.bupdate.update(self.b, db)

        return dx
    
    def getparams(self):
        return self.w,self.b
    
    def setparams(self, w,b):
        self.w=w
        self.b=b
    
    


# In[3]:


class relu(object):
    def layername(self):
        return "RELU"  
    
    def forward(self,x):
        out = np.maximum(x, 0)
        self.x = x
        return out

    def backward(self,dout):
        dx = np.where(self.x < 0, 0, dout)
        #dx[x<0] = 0
        return dx


# In[4]:


class norm(object):
    def __init__(self, gamma, beta,bn_param):
        self.gamma=gamma
        self.beta=beta
        self.bn_param=bn_param
        self.eps=1e-5;
        self.momentum=0.9;
        
    def layername(self):
        return "normalizaion"  
        
    def forward(self,x):
        eps=self.eps
        momentum=self.momentum
        bn_param=self.bn_param   
        mode = bn_param['mode']
        N, D = x.shape
        running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
        running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

        out, cache = None, None
        if mode == 'train':
        
            N = float(N)
            mu = 1 / N * np.sum(x, axis=0)
            xmu = x - mu
            carre = xmu ** 2
            var = 1 / N * np.sum(carre, axis=0)
            sqrtvar = np.sqrt(var + eps)
            invvar = 1. / sqrtvar
            va2 = xmu * invvar
            va3 = self.gamma * va2
            out = va3 + self.beta
        
            running_mean = momentum * running_mean + (1.0 - momentum) * mu
            running_var = momentum * running_var + (1.0 - momentum) * var
        
            self.cache = (mu, xmu, carre, var, sqrtvar, invvar,
                        va2, va3, self.gamma, self.beta, x, bn_param)
        elif mode == 'test':

            sample_mean = running_mean
            sample_var = running_var
            out = (x- sample_mean)/(np.sqrt(sample_var + eps))
            out = self.gamma * out + self.beta

        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var
        
        self.bn_param=bn_param

        return out


    def backward(dout):

        dx=None
        mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = self.cache
        eps = self.eps
        N, D = dout.shape
        N = float(N)
        dva3 = dout
        self.beta = np.sum(dout, axis=0)
        dva2 = gamma * dva3
        self.gamma = np.sum(va2 * dva3, axis=0)
        dxmu = invvar * dva2
        dinvvar = np.sum(xmu * dva2, axis=0)
        dsqrtvar = -1./(sqrtvar ** 2) * dinvvar
        dvar = 0.5 * (var + eps) ** (-0.5) * dsqrtvar
        dcarre = 1/N * np.ones(carre.shape) * dvar
        dxmu += 2 * xmu * dcarre
        dx = dxmu
        dmu = - np.sum(dxmu, axis=0)
        dx += 1/N * np.ones(dxmu.shape) * dmu

        return dx
    
    def getparams(self):
        return self.gamma,self.beta
    
    def setparams(self,g,b):
        self.gamma=w
        self.beta=b


# In[ ]:


class lossfunc(object):
    def forward (self,X,y_train):
        scores = X
        probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs /= np.sum(probs, axis = 1, keepdims = True)
        data_loss = -np.sum(np.log(probs[np.arange(scores.shape[0]), y_train]))/scores.shape[0]

        #get derivative of scores
        deriv_scores = probs.copy()
        deriv_scores[np.arange(scores.shape[0]), y_train] -= 1
        deriv_scores /= scores.shape[0] 
        
        return data_loss, deriv_scores 
    
    def layername(self):
        return "softmax"

        

# In[ ]:


class adam(object):
    def __init__(self, inp):
        self.learning_rate = 1e-3
        self.beta1 = 0.9
        self.beta2 = 0.999 
        self.epsilon =  1e-8
        self.m = np.zeros_like(inp)
        self.v= np.zeros_like(inp)
        self.t = 0

    
    def update (self,x, dx):
        self.t = self.t + 1
        next_x = None
        self.m =  self.beta1* self.m + (1- self.beta1)*dx
        self.v =  self.beta2* self.v + (1- self.beta2)*(dx**2)
        mb =  self.m/(1 -  self.beta1 **  self.t)
        vb =  self.v/(1 -  self.beta2 **  self.t)
        next_x = x -  self.learning_rate * mb / (np.sqrt(vb) +  self.epsilon)

        return next_x

