import numpy as np
import Pipeline
import pandas as pd
from matplotlib import pyplot as plt
import h5py
class Model:

    def __init__(self, L1size, L2size, L3Size, weights=""):
       
        self.W1 = np.zeros(L1size, L2size)    
        self.B1 = np.zeros(1,L2size)
        self.W2 = np.zeros(L2size, L3Size)
        self.B2 = np.zeros(1,L3Size)
        self.W3 = np.zeros(L3Size,1)
        self.B3 = np.zeros(1,1)

        if len(weights) > 0:
            self.load_weights(weights)

    def forward(self, x):
        X = np.array(x).astype(float)
        #multiply input by W1
        L1= X @ self.W1
        #+ self.B1
        #Input L1 into activation function
        L1a = self.ReLU(L1)
        #Multiply L1a by W2 matrix
        L2 = L1a @ self.W2 
        #+ self.B2
        L2a = self.ReLU(L2)
        L3 = L2a @ self.W3
        # + self.B3
        yHat = self.sigmoid(L3)

        return yHat, L3, L2a, L2, L1a, L1
    
    def predict(self, X):
 
        X = np.array(X).astype(float)
        #multiply input by W1
        L1= X @ self.W1 + self.B1
        #Input L1 into activation function
        L1a = self.ReLU(L1)
        #Multiply L1a by W2 matrix
        L2 = L1a @ self.W2 + self.B2
        L2a = self.ReLU(L2)
        L3 = L2a @ self.W3 + self.B3
        yHat = self.sigmoid(L3)

        return yHat
    
    def compute_gradient(self, X, y, yHat, L3, L2a, L2, L1a, L1):
        
        #derivative of cost function
        dJdY = self.BCE_Prime(y,yHat)
        #derivative with respect to W3
        dJdW3 = np.matmul(np.asmatrix(L2a).T,dJdY * self.sigmoidPrime(L3))
        #derivative with respect to B3
        dJdB3 = dJdY * self.sigmoidPrime(L3)
        #derivative with respect to W2
        dJdW2 = np.matmul(np.asmatrix(L1a).T,np.matmul(np.multiply(dJdY,self.sigmoidPrime(L3)),np.multiply(np.asmatrix(self.W3).T,self.ReLUPrime(L2))))
        #derivative with respect to B2
        dJdB2 = np.matmul(np.multiply(dJdY,self.sigmoidPrime(L3)),np.multiply(np.asmatrix(self.W3).T,self.ReLUPrime(L2)))
        #derivative with respect to W1
        dJdW1 = np.matmul(np.asmatrix(X).T,np.matmul(np.matmul(np.multiply(dJdY,self.sigmoidPrime(L3)),np.multiply(np.asmatrix(self.W3).T,self.ReLUPrime(L2))),np.multiply(np.asmatrix(self.W2).T,self.ReLUPrime(L1))))
        #derivative with respect to B1
        dJdB1 = np.matmul(np.matmul(np.multiply(dJdY,self.sigmoidPrime(L3)),np.multiply(np.asmatrix(self.W3).T,self.ReLUPrime(L2))),np.multiply(np.asmatrix(self.W2).T,self.ReLUPrime(L1)))
        
        return dJdW3, dJdB3, dJdW2, dJdB2, dJdW1, dJdB1
        
        

    """""
    def compute_gradient(self, X, y, yHat, L3, L2a, L2, L1a, L1): 
        
        dJdY = self.BCE_Prime(y, yHat)
        #derivative with respect to W3
        dJdW3 = np.asmatrix(L2a).T @ (dJdY * self.sigmoidPrime(L3))
        #derivative with respect to B3
        dJdB3 = dJdY * self.sigmoidPrime(L3)
        #derivative with respect to W2
        dJdW2 = np.asmatrix(L1a).T @ np.multiply((dJdY * self.sigmoidPrime(L3)) @ np.asmatrix(self.W3).T,self.ReLUPrime(L2))
        #derivative with respect to B2
        dJdB2 = np.multiply((dJdY * self.sigmoidPrime(L3)) @ np.asmatrix(self.W3).T,self.ReLUPrime(L2))
        #derivative with respect to W1
        dJdW1 = np.asmatrix(X).T @ np.multiply(np.multiply((dJdY * self.sigmoidPrime(L3)) @ np.asmatrix(self.W3).T,self.ReLUPrime(L2)) @ np.asmatrix(self.W2).T, self.ReLUPrime(L1)) 
        #derivative with respect to B1
        dJdB1 = np.multiply(np.multiply((dJdY * self.sigmoidPrime(L3)) @ np.asmatrix(self.W3).T,self.ReLUPrime(L2)) @ np.asmatrix(self.W2).T, self.ReLUPrime(L1))
        return dJdW3, dJdB3, dJdW2, dJdB2, dJdW1, dJdB1
    """

    def batch_average(self, dJdW3, dJdB3, dJdW2, dJdB2, dJdW1, dJdB1, batch_size):

        if(batch_size>1):
            dJdW3 = np.sum(dJdW3,axis=0) * (1/batch_size) 
            dJdW2 = np.sum(dJdW2,axis=0) * (1/batch_size) 
            dJdW1 = np.sum(dJdW1,axis=0) * (1/batch_size) 

            dJdB3 = np.sum(dJdB3) * (1/batch_size) 
            dJdB2 = np.sum(dJdB2,axis=0) * (1/batch_size) 
            dJdB1 = np.sum(dJdB1,axis=0) * (1/batch_size)
        return dJdW3, dJdB3, dJdW2, dJdB2, dJdW1, dJdB1
    
    def updateSGD(self, dJdW3, dJdB3, dJdW2, dJdB2, dJdW1, dJdB1, alpha):

        self.W3 -= alpha * dJdW3
        self.W2 -= alpha * dJdW2
        self.W1 -= alpha * dJdW1
        self.B3 -= alpha * dJdB3
        self.B2 -= alpha * dJdB2
        self.B1 -= alpha * dJdB1

    def SGD(self, data, alpha=.05, split=.7, batch_size=1):
        data = pd.read_csv(data, index_col=0)
        for i in range(int(data.shape[1]*split/batch_size)):
            X = np.array(data.iloc[0:data.shape[0]-1,i:i+batch_size]).T
            y = np.array(data.iloc[data.shape[0]-1,i:i+batch_size])
            yHat, L3, L2a, L2, L1a, L1 = self.forward(X)
            dJdW3, dJdB3, dJdW2, dJdB2, dJdW1, dJdB1 = self.compute_gradient(X,y, yHat, L3, L2a, L2, L1a, L1)           
            dJdW3, dJdB3, dJdW2, dJdB2, dJdW1, dJdB1 = self.batch_average(dJdW3, dJdB3, dJdW2, dJdB2, dJdW1, dJdB1, batch_size)
            self.updateSGD(dJdW3, dJdB3, dJdW2, dJdB2, dJdW1, dJdB1, alpha)
        X = np.array(data.iloc[0:data.shape[0]-1,int(data.shape[1]*split):data.shape[1]]).T
        y = np.array(data.iloc[data.shape[0]-1,int(data.shape[1]*split):data.shape[1]])
        yHat = self.threshHold(self.predict(X))
        accuracy = 0
        for i in range(len(y)):
            if y[i] == yHat[i]:
                accuracy +=1
        accuracy/=len(y)
        print("Accuracy: " + str(accuracy*100)+"%")
        if input("Save weights (y/n)\n").upper().strip(" ") == "Y":
            self.save_weights()

    def ADAM(self, data, alpha=.001, beta1=.9, beta2=.999, epsilon=.000000001, split=.8,batch_size=10):
        data = pd.read_csv(data, index_col=0)
        self.mW1 = np.zeros(self.W1.shape)    
        self.mB1 = np.zeros(self.B1.shape)
        self.mW2 = np.zeros(self.W2.shape)
        self.mB2 = np.zeros(self.B2.shape)
        self.mW3 = np.zeros(self.W3.shape)
        self.mB3 = np.zeros(self.B3.shape)

        self.vW1 = np.zeros(self.W1.shape)    
        self.vB1 = np.zeros(self.B1.shape)
        self.vW2 = np.zeros(self.W2.shape)
        self.vB2 = np.zeros(self.B2.shape)
        self.vW3 = np.zeros(self.W3.shape)
        self.vB3 = np.zeros(self.B3.shape)

        for i in range(int(data.shape[1]*split/batch_size)):
            X = np.array(data.iloc[0:data.shape[0]-1,i:i+batch_size]).T
            y = np.array(data.iloc[data.shape[0]-1,i:i+batch_size])
            yHat, L3, L2a, L2, L1a, L1 = self.forward(X)
            dJdW3, dJdB3, dJdW2, dJdB2, dJdW1, dJdB1 = self.compute_gradient(X,y, yHat, L3, L2a, L2, L1a, L1)
            dJdW3, dJdB3, dJdW2, dJdB2, dJdW1, dJdB1 = self.batch_average(dJdW3, dJdB3, dJdW2, dJdB2, dJdW1, dJdB1,batch_size)
            self.firstMoment(dJdW3, dJdB3, dJdW2, dJdB2, dJdW1, dJdB1, beta1)
            self.secondMoment(dJdW3, dJdB3, dJdW2, dJdB2, dJdW1, dJdB1, beta2)
            self.biasCorrect(beta1,beta2)
            self.update(alpha,epsilon)
            

        X = np.array(data.iloc[0:data.shape[0]-1,int(data.shape[1]*split):data.shape[1]]).T
        y = np.array(data.iloc[data.shape[0]-1,int(data.shape[1]*split):data.shape[1]])
        yHat = self.threshHold(self.predict(X))
        accuracy = 0
        for i in range(len(y)):
            if y[i] == yHat[i]:
                accuracy +=1
        accuracy/=len(y)
        print("Accuracy: " + str(accuracy*100)+"%")
        if input("Save weights (y/n)\n").upper().strip(" ") == "Y":
            self.save_weights()
    
    def firstMoment(self, dJdW3, dJdB3, dJdW2, dJdB2, dJdW1, dJdB1, beta1):
        self.mW1 = beta1 * self.mW1 + (1-beta1) * dJdW1   
        self.mB1 = beta1 * self.mB1 + (1-beta1) * dJdB1
        self.mW2 = beta1 * self.mW2 + (1-beta1) * dJdW2
        self.mB2 = beta1 * self.mB2 + (1-beta1) * dJdB2
        self.mW3 = beta1 * self.mW3 + (1-beta1) * dJdW3
        self.mB3 = beta1 * self.mB3 + (1-beta1) * dJdB3
    
    def secondMoment(self, dJdW3, dJdB3, dJdW2, dJdB2, dJdW1, dJdB1, beta2):
        self.vW1 = beta2 * self.vW1 + (1-beta2) * np.square(dJdW1)
        self.vB1 = beta2 * self.vB1 + (1-beta2) * np.square(dJdB1)
        self.vW2 = beta2 * self.vW2 + (1-beta2) * np.square(dJdW2)
        self.vB2 = beta2 * self.vB2 + (1-beta2) * np.square(dJdB2)
        self.vW3 = beta2 * self.vW3 + (1-beta2) * np.square(dJdW3)
        self.vB3 = beta2 * self.vB3 + (1-beta2) * np.square(dJdB3)
    
    def biasCorrect(self,beta1,beta2):
        self.CmW1 = self.mW1 / (1-beta1)  
        self.CmB1 = self.mB1 / (1-beta1)
        self.CmW2 = self.mW2 / (1-beta1)
        self.CmB2 = self.mB2 / (1-beta1) 
        self.CmW3 = self.mW3 / (1-beta1)
        self.CmB3 = self.mB3 / (1-beta1)
        
        self.CvW1 = self.vW1 / (1-beta2)  
        self.CvB1 = self.vB1 / (1-beta2)
        self.CvW2 = self.vW2 / (1-beta2)
        self.CvB2 = self.vB2 / (1-beta2) 
        self.CvW3 = self.vW3 / (1-beta2)
        self.CvB3 = self.vB3 / (1-beta2)

    def update(self,alpha,epsilon):
        self.W1 -= alpha * np.divide(self.CmW1,np.sqrt(self.CvW1+epsilon))
        self.W2 -= alpha * np.divide(self.CmW2,np.sqrt(self.CvW2+epsilon))
        self.W3 -= alpha * np.divide(self.CmW3,np.sqrt(self.CvW3+epsilon))

        self.B3 -= alpha * np.divide(self.CmB3,np.sqrt(self.CvB3+epsilon))
        self.B2 -= alpha * np.divide(self.CmB2,np.sqrt(self.CvB2+epsilon))
        self.B1 -= alpha * np.divide(self.CmB1,np.sqrt(self.CvB1+epsilon))

    def GELU(self, x):
        return 0.5*x*(1+self.tanh(0.7978*(x+.044715*x**3)))
    
    def GELUPrime(self,x):
        return .5*(self.tanh((.7978*(x+x**3)))+x*self.tanhPrime(.7978*(x+x**3))*(1+3*x**2))
    
    def BCE(self,y):
        return -(y*np.log(self.yHat) + (1-y)*np.log(1-self.yHat))

    def BCE_Prime(self,y, yHat):
        y = y.reshape(y.shape[0],1)
        y = yHat.reshape(yHat.shape[0],1)
        return -1*(yHat-y)/(np.square(yHat)-yHat + .000000001)
    
    def tanh(self,x):        
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def tanhPrime(self, x):
        return 1 - np.power(self.tanh(x), 2)
     
    def threshHold(self, vector):
        return [1 if probability >= .5 else 0 for probability in vector]
    
    def loss(self,y):
        return np.exp((y-self.yHat)**2)
    
    def SSE(self, y):
        return (y-self.yHat)**2
    
    def SSE_Prime(self, y):
        return 2*(y-self.yHat)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def ReLU(self,x):
        return np.maximum(0,x)
    
    def ReLUPrime(self,x): 
        return 1 * (x>0)

    def sigmoidPrime(self, x):
        return self.sigmoid(x) - (self.sigmoid(x) ** 2)
    
    def save_weights(self):
        response = input("Would you like to save the parameters from this training: (y/n)")
        if response.upper() == 'Y':
            with h5py.File("weights.hdf5", "w") as file:
                file.create_dataset("w1", data=self.W1)
                file.create_dataset("w2", data=self.W2)
                file.create_dataset("w3", data=self.W3)
                file.create_dataset("b1", data=self.B1)
                file.create_dataset("b2", data=self.B2)
                file.create_dataset("b3", data=self.B3)
            
    def load_weights(self, weights):
        with h5py.File(weights, "r") as file:
            self.W1 = np.array(file["w1"])
            self.W2 = np.array(file["w2"])
            self.W3 = np.array(file["w3"])

    def training_report(self,errors):    
        y= np.array(errors)
        x = np.arange(0, y.size)
        

        plt.plot(x,y,color="red")
        plt.show()
        self.save_weights()

   
"""""
    def train(self, data, learning_rate):
        data = pd.read_csv(data,index_col=0)
        error = []
        right = 0

        for i in range(data.shape[1]):
           
            X = np.array(data.iloc[:,i])
            y = X[X.size-1]
            X = np.delete(X, X.size-1)
            prediction = self.forward(X)
            if i >2000 and y==self.threshHold(prediction):
                right+=1
            print("Epoch: " + str(i) + " yHat: " + str(prediction) + " y: " + str(y))
            self.compute_gradient(X, y, yHat, L3, L2a, L2, L1a, L1)
            self.backProp(learning_rate)
            error.append(self.BCE(y))
        print("Accuracy: " + str(right/1028))
        self.training_report(error)
"""
            
"""""
    def mb_backprop(self, dJdW3, dJdB3, dJdW2, dJdB2, dJdW1, dJdB1, learning_rate,batch_size):

        self.W3 -= np.sum(dJdW3,axis=1) * (1/batch_size) * learning_rate
        self.W2 -= np.sum(dJdW2,axis=0) * (1/batch_size) * learning_rate
        self.W1 -= np.sum(dJdW1,axis=0) * (1/batch_size) * learning_rate

        self.B3 -= np.sum(dJdB3) * (1/batch_size) * learning_rate
        self.B2 -= np.sum(dJdB2,axis=0) * (1/batch_size) * learning_rate
        self.B1 -= np.sum(dJdB1,axis=0) * (1/batch_size) * learning_rate
"""  






