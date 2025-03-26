import math 
import numpy as np 
from Layer import * 
from sklearn.utils import shuffle 
from LROptimizerType import LROptimizerType 
from BatchNormMode import BatchNormMode  

class Network(object):     
    def __init__(self,X,Y,numLayers,batchsize,dropOut = 1.0,activationF=ActivationType.SIGMOID, lastLayerAF= ActivationType.SIGMOID):         
        self.X = X         
        self.Y = Y         
        self.batchsize = batchsize         
        self.numLayers = numLayers         
        self.Layers = []  # network contains list of layers         
        self.lastLayerAF = lastLayerAF         
        for i in range(len(numLayers)):             
            if (i == 0):  # first layer                 
                layer =  Layer(numLayers[i],X.shape[1],batchsize,False,dropOut, activationF)              
            elif (i == len(numLayers)-1):  # last layer                 
                layer =  Layer(Y.shape[1],numLayers[i-1],batchsize,True,dropOut, lastLayerAF)              
            else:  # intermediate layers                 
                layer = Layer(numLayers[i],numLayers[i-1],batchsize,False,dropOut, activationF)              
            self.Layers.append(layer); 

    def Evaluate(self,indata,doBatchNorm=False,batchMode=BatchNormMode.TEST):  # evaluates all layers         
         self.Layers[0].Evaluate(indata, doBatchNorm,batchMode)  # first layer         
         for i in range(1,len(self.numLayers)):             
             self.Layers[i].Evaluate(self.Layers[i-1].a,doBatchNorm,batchMode)         
         return self.Layers[len(self.numLayers)-1].a      
     
    def Train(self, epochs,learningRate, lambda1, batchsize=1, LROptimization=LROptimizerType.NONE,doBatchNorm=False):         
        itnum = 0         
        for j in range(epochs):             
            error = 0             
            self.X, self.Y = shuffle(self.X, self.Y, random_state=0)             
            for i in range(0, self.X.shape[0], batchsize):                 
                # get (X, y) for current minibatch/chunk                 
                X_train_mini = self.X[i:i + batchsize]                 
                y_train_mini = self.Y[i:i + batchsize]                  
                
                self.Evaluate(X_train_mini,doBatchNorm,batchMode=BatchNormMode.TRAIN)                 
                if (self.lastLayerAF == ActivationType.SOFTMAX):                     
                    error += -(y_train_mini * np.log(self.Layers[len(self.numLayers)-1].a+0.001)).sum()                 
                else:                     
                    error += ((self.Layers[len(self.numLayers)-1].a - y_train_mini) * (self.Layers[len(self.numLayers)-1].a - y_train_mini)).sum()                  
                
                lnum = len(self.numLayers)-1  # last layer number                  
                
                # compute deltas, grads on all layers                 
                while(lnum >= 0):                     
                    if (lnum == len(self.numLayers)-1):  # last layer                         
                        if (self.lastLayerAF == ActivationType.SOFTMAX):                             
                            self.Layers[lnum].delta = -y_train_mini + self.Layers[lnum].a                         
                        else:                             
                            self.Layers[lnum].delta = -(y_train_mini-self.Layers[lnum].a) * self.Layers[lnum].derivAF                     
                    else: # intermediate layer                         
                        self.Layers[lnum].delta = np.dot(self.Layers[lnum+1].delta,self.Layers[lnum+1].W) * self.Layers[lnum].derivAF                     
                    if (doBatchNorm == True):                         
                        self.Layers[lnum].dbeta = np.sum(self.Layers[lnum].delta,axis=0)                         
                        self.Layers[lnum].dgamma = np.sum(self.Layers[lnum].delta * self.Layers[lnum].Shat,axis=0)                         
                        self.Layers[lnum].deltabn = (self.Layers[lnum].delta * self.Layers[lnum].gamma)/(batchsize*np.sqrt(self.Layers[lnum].sigma2 +self.Layers[lnum].epsilon )) * (batchsize -1 - (self.Layers[lnum].Shat * self.Layers[lnum].Shat))                              
                    if (lnum > 0):  #previous output                         
                        prevOut = self.Layers[lnum-1].a                     
                    else:                         
                        prevOut = X_train_mini 
                    if (doBatchNorm == True):                         
                        self.Layers[lnum].WGrad = np.dot(self.Layers[lnum].deltabn.T,prevOut)                         
                        self.Layers[lnum].bGrad = self.Layers[lnum].deltabn.sum(axis=0)                     
                    else:                         
                        self.Layers[lnum].WGrad = np.dot(self.Layers[lnum].delta.T,prevOut)                         
                        self.Layers[lnum].bGrad = self.Layers[lnum].delta.sum(axis=0)                     
                    lnum = lnum - 1                  
                    
                itnum = itnum + 1                 
                self.UpdateGradsBiases(learningRate,lambda1, batchsize, LROptimization, itnum, doBatchNorm)              
            print("Iter = " + str(j) + " Error = "+ str(error))      
    
    def UpdateGradsBiases(self, learningRate, lambda1, batchSize, LROptimization, itnum, doBatchNorm):  
        # update weights and biases for all layers         
        beta1 = 0.9         
        beta2 = 0.999         
        epsilon = 1e-8         
        for ln in range(len(self.numLayers)):             
            if (LROptimization == LROptimizerType.NONE):                 
                self.Layers[ln].W = self.Layers[ln].W - learningRate * (1/batchSize) * self.Layers[ln].WGrad  - learningRate * lambda1 * self.Layers[ln].W.sum()                 
                self.Layers[ln].b = self.Layers[ln].b - learningRate * (1/batchSize) * self.Layers[ln].bGrad             
            elif (LROptimization == LROptimizerType.ADAM):                 
                gtw = self.Layers[ln].WGrad # weight gradients                 
                gtb = self.Layers[ln].bGrad # bias gradients                 
                self.Layers[ln].mtw = beta1 * self.Layers[ln].mtw + (1 - beta1) * gtw                 
                self.Layers[ln].mtb = beta1 * self.Layers[ln].mtb + (1 - beta1) * gtb                 
                self.Layers[ln].vtw = beta2 * self.Layers[ln].vtw + (1 - beta2) * gtw*gtw                 
                self.Layers[ln].vtb = beta2 * self.Layers[ln].vtb + (1 - beta2) * gtb*gtb                 
                mtwhat = self.Layers[ln].mtw/(1 - beta1**itnum)                 
                mtbhat = self.Layers[ln].mtb/(1 - beta1**itnum)                 
                vtwhat = self.Layers[ln].vtw/(1 - beta2**itnum)                 
                vtbhat = self.Layers[ln].vtb/(1 - beta2**itnum)                 
                self.Layers[ln].W = self.Layers[ln].W - learningRate * (1/batchSize) * mtwhat /((vtwhat**0.5) + epsilon)                 
                self.Layers[ln].b = self.Layers[ln].b - learningRate * (1/batchSize) * mtbhat /((vtbhat**0.5) + epsilon)               
                
            if (doBatchNorm == True):                 
                self.Layers[ln].beta = self.Layers[ln].beta - learningRate * self.Layers[ln].dbeta                 
                self.Layers[ln].gamma = self.Layers[ln].gamma - learningRate * self.Layers[ln].dgamma 