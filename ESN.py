import numpy as np
import random
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
        Given an initial probability, creates all weight matrix. Can also be used as a reinitialisation of the ESN.
        Idea of upgrade: give the initialisation of weigth as an argument, so that it is easily modulable.
        '''
        self.W = np.zeros((self.N,self.N))  #The internal weigth matrix
        self.W_in = 2*np.random.rand(self.N,self.number_input_) - 1  #We initialise between -1 and 1 uniformly, maybe to change
        self.W_out = 2*np.random.rand(self.N,self.number_input_) - 1
        self.W_fdb = 2*np.random.rand(self.N,self.number_input_) - 1

        #The loop will connect the reservoir with a certain probability.
        for i in range(self.N):
            for j in range(self.N):
                is_connected = (random.random()<proba_connexion)
                if is_connected:
                    self.W[i,j] = 2*random.random()-1   #Uniformly between -1 and 1, maybe to change.too

    def update(self,input = []):
        '''
        '''
        if input = []:
            input = np.zeros((self.number_input_))
        pass

    def simulation(self, nb_iter,initial_inputs):
        pass

    def save(self):
        pass

    @classmethod
    def load(self):
        pass

test= ESN(10,0.1,100,50)
print(test.W)
