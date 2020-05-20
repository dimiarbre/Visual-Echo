import numpy as np

class ESN:
    '''
    Notes that this is a specific Echo State Network for training purpose, without the maximum features.
    It may ultimately be a basic one for spatialisation purpose.
    '''
    def __init__(self,number_neurons, proba_connexion,number_input,number_output):
        '''
        Creates an instance of ESN given some parameters
        '''
        self.number_input_ = number_input
        self.number_output_ = number_output
        self.N = number_neurons  #How many neurons in our reservoir
        self.intern_activation = np.zeros((self.N))
        self.init_weights(proba_connexion)


    def init_weights(self,proba_connexion):
        '''
        Given an initial probability, creates all weight matrix. Can also be used as a reinitialisation of the ESN
        '''
        self.W = np.zeros((self.N,self.N))  #The internal weigth matrix
        self.W_in = 2*np.random.rand(self.N,self.number_input_)-1  #We initialise between -1 and 1 (To review?)
        self.W_out = 2*np.random.rand(self.N,self.number_input_)-1
        self.W_fdb = 2*np.random.rand(self.N,self.number_input_)-1

    def update(self,input = np.zeros(())):
        '''
        '''
        pass


test= ESN(10,1,100,50)
print(test.W_in)
