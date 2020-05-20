import numpy as np
import random

##Basic functions, used in this particular ESN class. Please note that since it is not a general one, they aren't modulable in this case, and changes has to be done by hand.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    expo = np.exp(-2*x)
    return (1-expo)/(1+expo)


class ESN:
    '''
    Notes that this is a specific Echo State Network for training purpose, without the maximum features.
    It may ultimately be a basic one for spatialisation purpose.
    '''
    def __init__(self,number_neurons, proba_connexion,number_input,number_output,spectral_radius):
        '''
        Creates an instance of ESN given some parameters
        :parameters:
            - number_neurons : The number of neurons in the reservoir
            - proba_connexion : the probability of connecting to another neuron (or the density of connexion in the reservoir)
            - number_input : How many input neurons
            - number_output : How many output neurons
            - spectral_radius : the desired spectral radius, depending on how lastong we want the memory to be.
        '''
        self.number_input = number_input
        self.number_output = number_output
        self.N = number_neurons  #How many neurons in our reservoir
        self.x = np.zeros((self.N))   #Intern state of the reservoir. Initialisation might change
        self.y = np.zeros((self.number_output))
        self.n_iter = 0

        self.W = np.zeros((self.N,self.N))  #The internal weigth matrix
        #The loop will connect the reservoir with a certain probability.
        for i in range(self.N):
            for j in range(self.N):
                is_connected = (random.random()<proba_connexion)
                if is_connected:
                    self.W[i,j] = 2*random.random()-1   #Uniformly between -1 and 1, maybe to change
        eigenvalue = max(abs(np.linalg.eig(self.W)[0]))
        print(eigenvalue,self.W)
        self.W *= spectral_radius/eigenvalue            #We normalize the weight matrix to get the desired spectral radius.
        self.W_in = 2*np.random.rand(self.N,self.number_input) - 1  #We initialise between -1 and 1 uniformly, maybe to change
        self.W_out = 2*np.random.rand(self.number_output, self.number_output+self.N + self.number_input) - 1
        self.W_back = 2*np.random.rand(self.N,self.number_output) - 1 #The Feedback matrix

    def update(self,input = []):
        '''
        Advance the process by 1 step
        '''
        if input == []:
            input = np.zeros((self.number_input))
        self.x = sigmoid(np.dot(self.W_in, input) + np.dot(self.W, self.x) + np.dot(self.W_back, self.y))       #Sigmoid use
        tab1 = np.concatenate((input,self.x))
        tab2 = np.concatenate((tab1,self.y))
        self.y = tanh(np.dot(self.W_out,tab2))                   #We use tanh for the output.
        self.n_iter +=1


    def simulation(self, nb_iter,initial_inputs):
        '''
        Simulates the behaviour of the ESN given a starting sequence (initial_inputs) for nb_iter iterations.
        '''
        for input in initial_inputs:
            self.update(input)
        self.n_iter = 0     #We reset the iterations, and should have an initialised reservoir.
        while self.n_iter <= nb_iter:
            self.update()


    def save(self,name):
        pass

    @classmethod
    def load(self,name):
        pass

test= ESN(number_neurons = 10,proba_connexion = 0.1,number_input = 100, number_output = 1,spectral_radius = 0.7)
print(max(abs(np.linalg.eig(test.W)[0]))) #Check wether the spectral radius is respected.
print(test.y)
test.update()
print(test.y)
test.update()
print(test.y)
