import numpy as np 
from ActivationType import ActivationType 
from BatchNormMode import BatchNormMode 

class Layer(object): 
    def __init__(self,numNeurons,numNeuronsPrevLayer, batchsize,lastLayer= False,dropOut = 0,activationType=ActivationType.SIGMOID): 
        # initialize the weights and biases 
        self.numNeurons = numNeurons 
        self.batchsize = batchsize 
        self.lastLayer = lastLayer 
        self.numNeuronsPrevLayer = numNeuronsPrevLayer 
        self.activationFunction = activationType 
        self.dropOut = dropOut 
        self.W = np.random.uniform(low=0.1,high=0.1,size=(numNeurons,numNeuronsPrevLayer))      
        self.b = np.random.uniform(low=-1,high=1,size=(numNeurons))  
        self.WGrad = np.zeros((numNeurons,numNeuronsPrevLayer)) 
        self.bGrad = np.zeros((numNeurons)) # gradient for delta 
        
        #-------------following for batch norm--------------
        self. mu = np.zeros((numNeurons))  # batch mean
        self.sigma2 = np.zeros((numNeurons)) # sigma^2 for batch
        self.epsilon = 1e-6
        self.gamma = np.random.rand(1)  
        self.beta= np.random.rand(1) 
        self.S = np.zeros((batchsize,numNeurons)) 
        self.Shat = np.zeros((batchsize,numNeurons))         
        self.Sb = np.zeros((batchsize,numNeurons))         
        self.runningmu = np.zeros((numNeurons))         
        self.runningsigma2 = np.zeros((numNeurons))         
        self.dgamma = np.zeros((numNeurons))         
        self.dbeta = np.zeros((numNeurons))         
        self.delta = np.zeros((batchsize,numNeurons))         
        self.deltabn = np.zeros((batchsize,numNeurons))         
        #---------------------------------------------------                  
        
        #-------------Implementing ADAM------------------         
        self.mtw = np.zeros((numNeurons,numNeuronsPrevLayer))         
        self.mtb = np.zeros((numNeurons))         
        self.vtw = np.zeros((numNeurons,numNeuronsPrevLayer))         
        self.vtb = np.zeros((numNeurons))         
        
        #----------------------------------------------------        
        self.zeroout = None  # for dropout         
        
    def Evaluate(self,indata, doBatchNorm=False, batchMode= BatchNormMode.TRAIN):         
        self.S = np.dot(indata,self.W.T) + self.b         
        if (doBatchNorm == True):             
            if (batchMode == BatchNormMode.TRAIN):                 
                self.mu = np.mean(self.S, axis=0)   # batch mean                 
                self.sigma2 = np.var(self.S,axis=0) # batch sigma^2                 
                self.runningmu = 0.9 * self.runningmu + (1 - 0.9)* self.mu                  
                self.runningsigma2 = 0.9 * self.runningsigma2 + (1 - 0.9)* self.sigma2             
            else:                 
                self.mu = self.runningmu                 
                self.sigma2 = self.runningsigma2             
            self.Shat = (self.S - self.mu)/np.sqrt(self.sigma2 + self.epsilon)             
            self.Sb = self.Shat * self.gamma + self.beta             
            sum = self.Sb         
        else:             
            sum = self.S          
            
        if (self.activationFunction == ActivationType.SIGMOID):             
            self.a = self.sigmoid(sum)             
            self.derivAF = self.a * (1 - self.a)         
        if (self.activationFunction == ActivationType.TANH):             
            self.a = self.TanH(sum)             
            self.derivAF = (1 - self.a*self.a)         
        if (self.activationFunction == ActivationType.RELU):             
            self.a = self.Relu(sum)             
            #self.derivAF = 1.0 * (self.a > 0)             
            epsilon=1.0e-6             
            self.derivAF = 1. * (self.a > epsilon)             
            self.derivAF[self.derivAF == 0] = epsilon         
        if (self.activationFunction == ActivationType.SOFTMAX):             
            self.a = self.Softmax(sum)             
            self.derivAF = None  # delta computation for Softmax layer is in Network         
        if (self.lastLayer == False):             
            self.zeroout = np.random.binomial(1,self.dropOut,(self.numNeurons))/self.dropOut             
            self.a = self.a * self.zeroout 
            self.derivAF = self.derivAF * self.zeroout                   
            
    def sigmoid(self,x):         
        return 1 / (1 + np.exp(-x))  # np.exp makes it operate on entire array          
      
    def TanH(self, x):         
        return np.tanh(x)      
    
    def Relu(self, x):         
        return np.maximum(0,x)      
    
    def Softmax(self, x):
        if x.shape[0] == x.size:  # Handle 1D input (vector case)
            x_shifted = x - np.max(x)  # Subtract the max value to prevent overflow
            ex = np.exp(x_shifted)
            return ex / ex.sum()

        # Handle 2D input (matrix case)
        x_shifted = x - np.max(x, axis=1, keepdims=True)  # Subtract max along each row
        ex = np.exp(x_shifted)
        
        for i in range(ex.shape[0]):
            denom = ex[i, :].sum()  # Sum over each row
            ex[i, :] = ex[i, :] / denom  # Normalize each row
        
        return ex
