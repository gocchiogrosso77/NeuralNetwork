import numpy as np
import Pipeline
import pandas as pd
from matplotlib import pyplot as plt
import h5py
class Model:
#:)
    def __init__(self, Xsize, hlSize, weights=""):
       
        self.X = np.zeros(Xsize)
        self.W1 = np.random.randn(Xsize, hlSize)

        # try switching hlSize and 1 in the W2 parameters
        self.W2 = np.random.randn(hlSize, 1)
        self.b1 = np.random.rand()
        self.b2 = np.random.rand()
        if len(weights) > 0:
            self.load_weights(weights)





    def forward(self, x):
        self.X = np.array(x).astype(float)

        #multiply input by W1
    
        self.L1= np.matmul(self.X, self.W1)       
        #Input L1 into activation function
        self.L1a = self.ReLU(self.L1)
        #Multiply L1a by W2 matrix
        self.L2 = np.matmul(self.L1a ,self.W2)     
        #Input L2 into activation function
        self.yHat = self.sigmoid(self.L2)
        if self.yHat >= .5:
            self.yHat = 1
        else:
            self.yHat = 0
        return self.yHat
    
    def save_weights(self):

        response = input("Would you like to save the parameters from this training: (y/n)")
        if response.upper() == 'Y':
            with h5py.File("weights.hdf5", "w") as file:
                file.create_dataset("w1", data=self.W1)
                file.create_dataset("w2", data=self.W2)
            
    def load_weights(self, weights):
        with h5py.File(weights, "r") as file:
            self.W1 = np.array(file["w1"])
            self.W2 = np.array(file["w2"])

 
    
    def compute_gradient(self ,y):
        #derivative of loss function
        if y == 1:
            self.dJdY = 1/self.yHat
        else:
            self.dJdY = y/self.yHat + 1-self.yHat
            
        #derivative with respect to the W2 weight matrix
        self.dJdW2 = self.dJdY * np.matmul(self.sigmoidPrime(self.L2), self.L2.T)

        dJdW1 = np.matmul(self.dJdY * self.sigmoidPrime(self.L2).astype(float), self.W2.T, self.ReLUPrime(self.L1).astype(float))
        self.dJdW1 = np.matmul(np.asmatrix(self.X).T, np.asmatrix(dJdW1))

    def loss_function(self,y):
        return (y*np.log(self.yHat) + (1-y)*np.log(1-self.yHat))[0]


    def tanh(self,x):        
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def tanhPrime(self, x):
        return 1 - np.power(self.tanh(x), 2)
    
    def backProp(self, learning_rate):

        self.W2 -= self.dJdW2 * learning_rate        
        self.W1 -= self.dJdW1 * learning_rate
    
    def training_report(self,errors):
        
        y= np.array(errors)
        x = np.arange(0, y.size)
        

        plt.plot(x,y,color="red")
        plt.show()
        self.save_weights()




    #use sigmoid as opposed to softmax for binary classification
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def ReLU(self,x):
        return np.maximum(0,x)
    
    def ReLUPrime(self,x): 
        return 1 * (x>0)

    def sigmoidPrime(self, x):
        return self.sigmoid(x) - (self.sigmoid(x) ** 2)

    def train(self, data, learning_rate):
        data = pd.read_csv(data,index_col=0)
        print(data.head())
        error = []
        for i in range(data.shape[1]):
           
            X = np.array(data.iloc[:,i])
            y = X[X.size-1]
            X = np.delete(X, X.size-1)
            prediction = self.forward(X)
            print("Epoch: " + str(i) + " yHat: " + str(prediction) + " y: " + str(y))
            self.compute_gradient(y)
            self.backProp(learning_rate)
            error.append(self.loss_function(y))
        
        self.training_report(error)

            







