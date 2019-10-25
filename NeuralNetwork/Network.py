
# coding: utf-8

# In[5]:

import numpy as np
from Layers import *
from Layers2 import *



class network(object):
    def __init__(self,input_dim=3*32*32, num_classes=10,reg=0.1,
                             weight_scale=1e-2, dtype=np.float32, gamma=0.9,beta=0.1 ):
        self.layers=[]
        self.losses=[]
        self.trlosses=[]
        self.trainingacc=[]
        self.valiacc=[]
        self.num=0
        self.bestacc=0
        self.bestweights=[]
        
      
    def add(self,layer):
        self.layers.append(layer)
        self.num=self.num+1
        
    
    
    def step(self, X_minibatch, Y_minibatch):
        s = X_minibatch
        for i in range(self.num-1):
            s = self.layers[i].forward(s)
        
        loss, dx = self.layers[self.num-1].forward(s, Y_minibatch)
        predictions = np.argmax(s, axis=1)
        accuracy = np.mean(predictions == Y_minibatch)
        
        for i in range(self.num-2, -1, -1):
            dx = self.layers[i].backward(dx)
            
        return loss, accuracy

    def evaluate(self, X_val, Y_val, step_size = 100):                  
        N = X_val.shape[0]
        steps = N // step_size
        Losses = np.zeros(steps)
        Accuracies = np.zeros(steps)
        for k in range(steps):
            s = X_val[k * step_size : (k+1) * step_size]
            y = Y_val[k * step_size : (k+1) * step_size]
            for i in range(self.num-1):
                s = self.layers[i].forward(s)
            Losses[k], dx = self.layers[self.num-1].forward(s, y)
            predictions = np.argmax(s, axis=1)
            Accuracies[k] = np.mean(predictions == y)
        
        return np.mean(Losses), np.mean(Accuracies)

    
    def epoch(self, X_tr, Y_tr, X_val, Y_val, batch_size=100):
        N = X_tr.shape[0]
        batches = N // batch_size
        Losses = np.zeros(batches)
        Accuracies = np.zeros(batches)
        for k in range(batches):
            X_minibatch = X_tr[k * batch_size : (k+1) * batch_size]
            Y_minibatch = Y_tr[k * batch_size : (k+1) * batch_size]
            Losses[k], Accuracies[k] = self.step(X_minibatch, Y_minibatch)
        val_loss, val_acc = self.evaluate(X_val, Y_val, step_size = batch_size)
        tr_loss, tr_acc = np.mean(Losses), np.mean(Accuracies)
        
        self.trlosses.append(tr_loss)
        self.losses.append(val_loss)
        self.trainingacc.append(tr_acc)
        self.valiacc.append(val_acc)
        
        if(val_acc>self.bestacc):
            self.bestweights=[]
            self.bestacc=val_acc
            for i in range (self.num):
                if (self.layers[i].layername()=="conv" or self.layers[i].layername()=="layer") :
                    self.bestweights.append(self.layers[i].getparams())
                
                
        
    
    def train(self, X_tr, Y_tr, X_val, Y_val, num_epochs=50, batch_size=100):
        self.losses=[]
        self.trlosses=[]
        self.trainingacc=[]
        self.valiacc=[]
        
        for i in range(num_epochs):
            self.epoch(X_tr, Y_tr, X_val, Y_val, batch_size=batch_size)
            print("Epoch: %d tr_loss = %f val_loss = %f tr_acc = %f val_acc = %f\n"
                  % (i, self.trlosses[i], self.losses[i], self.trainingacc[i], self.valiacc[i]))
            
        c=0
        for i in range (self.num):
            if (self.layers[i].layername()=="conv" or self.layers[i].layername()=="layer") :
                self.layers[i].setparams(self.bestweights[c])
                c=c+1
