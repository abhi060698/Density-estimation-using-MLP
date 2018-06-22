
# coding: utf-8

# In[1]:


#import required header files
import keras
import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#import all the modules needed
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras import regularizers
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
model = Sequential()


# In[3]:


class ANN:
    
    def __init__(self,X,Y,nodes,model=model,input_activation='relu',hidden_layers_activation='relu',output_activation='softmax',loss='categorical_crossentropy',optimizer='adam',epochs=10,batch_size=200,kernel_init='normal'):#constructor
        self.X                          = [[X[i][j] for j in range(len(X[i]))] for i in range(len(X))]       
        self.Y                          = [Y[i] for i in range(len(Y))]                                      
        self.input_dim                  = nodes[0]                                                           
        self.nodes                      = [nodes[i] for i in range(len(nodes))]                              
        self.model                      = model                                                              
        self.layers                     = len(nodes)                                                                                      
        self.input_activation           = input_activation
        self.hidden_layers_activation   = hidden_layers_activation
        self.output_activation          = output_activation                                                                                     
        self.loss                       = loss                                                               
        self.optimizer                  = optimizer                                                     
        self.epochs                     = epochs                                                             
        self.batch_size                 = batch_size                                                         
        self.kernel_init                = kernel_init                                                       
                                                                    
        
        self.model.add(Dense(nodes[0],input_dim=self.input_dim,kernel_initializer=self.kernel_init,activation=self.input_activation))
       
        if(self.layers>2):
            for i in range(1,self.layers-1):
                self.model.add(Dense(self.nodes[i], kernel_initializer=self.kernel_init,activation=self.hidden_layers_activation))            
        
        self.model.add(Dense(self.nodes[(self.layers)-1], kernel_initializer=self.kernel_init,activation=self.output_activation))
        
        self.model.compile(loss=self.loss,optimizer=self.optimizer)
        print(model.summary())
        
        np.random.seed(7)
        self.model.fit(np.array(self.X),np.array(self.Y),epochs= self.epochs,batch_size=self.batch_size,verbose=2)
            
     
    def check_accuracy(self,P,Q):#check the accuracy of the trained model on test data
        scores               = self.model.evaluate(np.array(P), np.array(Q),verbose=0)
        print("\n%s: %.2f%%" % ("accuracy of the classifier on the validation dataset ", scores[1]*100))
        
    def test_code(self,P,Q):#check the accuracy of the trained model on test data
        np.random.seed(7)
        scores               = self.model.evaluate(np.array(P), np.array(Q),verbose=0)
        if(scores[1]>0.97):
            print("\n%s: %.2f%%" % ("accuracy of the classifier on the validation dataset ", scores[1]*100))
            print("code passes the test")
            
        else:
            print("code has failed the test")
            
        
        
    def predict(self,X_new):#make predictions for new feature data
        predictions = self.model.predict(X_new)
        return predictions
       
        
    def clear_model(self):
        from keras import backend as K
        K.clear_session()
        
    def save_model_and_weights(self):#save the model as a json file to disk
        #saving model and weights
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")
        
   

class Retrain:
    
    def __init__(self,json_file,output_dim,X,Y,epochs=10,batch_size=200):
        self.model          = model
        self.json_file      = json_file
        self.output_dim     = output_dim
        self.X              = [[X[i][j] for j in range(len(X[i]))] for i in range(len(X))]
        self.Y              = [Y[i] for i in range(len(Y))] 
        self.epochs         = epochs
        self.batch_size     = batch_size
        
        #loading the new model
        json_file = open(self.json_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model2= model_from_json(loaded_model_json)
        # load weights into new model
        model2.load_weights("model.h5")
        print("Loaded model from disk")
        
        self.model2.layers.pop()
        self.model2.add(Dense(self.output_dim,activation=activation))
        
        self.model.fit(np.array(self.X),np.array(self.Y),epochs= self.epochs,batch_size=self.batch_size,verbose=2)
        
        scores               = self.model.evaluate(np.array(self.X), np.array(self.Y))
        print("\n%s: %.2f%%" % ("accuracy of the incrementally trained classifier on the training dataset ", scores[1]*100))
            
        
    def check_accuracy(self,choice,P,Q):#check accuracy of the incrementally trained model 
        scores               = self.model2.evaluate(np.array(P), np.array(Q))
        print("\n%s: %.2f%%" % ("accuracy of the incrementally trained classifier on the validation dataset ", scores[1]*100))
        
       
    def predict(self,X_new):#make predictions using incrementally trained model
        predictions = self.model2.predict(X_new)
        df = X_new[:,0]
        df['predictions']=predictions
        df.to_csv('predictions2.csv')    


# In[4]:


def test(): #perform a test to see if the code works using the MNIST dataset
    from keras.datasets import mnist
    np.random.seed(7)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    y_train = np_utils.to_categorical(y_train,10)
    y_test = np_utils.to_categorical(y_test,10)
    num_classes = y_test.shape[0]
    c=ANN(X_train,y_train,[784,10])
    proceed=c.test_code(X_test,y_test)
    c.clear_model()
    return proceed


# In[5]:


print('''(self,X,Y,nodes=,model=model,activation='softmax',loss='categorical_crossentropy',optimizer='adam',
         epochs=10,batch_size=200,kernel_init='normal') are the parameters of the constructors for class ANN
         
         X,Y and input_dim are the essential parameters.The remaining parameters are all optional.
         
         An example of a call to the constructor is :      c = ANN(X,Y,20)
         
         However, if we want to replace certain defaults,  c = ANN(X,Y,20,optimixer='RMSProp')''' )

