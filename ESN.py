import numpy as np
import random
import matplotlib.pyplot as plt
##Basic functions, used in this particular ESN class. Please note that since it is not a general one, they aren't modulable in this case, and changes has to be done by hand.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    expo = np.exp(-2*x)
    return (1-expo)/(1+expo)

def reciprocal_tanh(x):
    return 0.5 * (np.ln(1 + x) - np.ln(1 - x))

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
        if eigenvalue == 0.0:
            raise Exception("Null Maximum Eigenvalue")
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

        '''
        Important: to change to leaky integrator.
        '''
        test= np.copy(self.x)
        matrix1 = np.dot(self.W_in, input)
        matrix2 = np.dot(self.W, self.x)
        matrix3 = np.dot(self.W_back, self.y)
        self.x = sigmoid( matrix1 + matrix2 + matrix3)       #Sigmoid use -> should change the function for discrete time
        if (np.array_equal(test,self.x)):
            print("Etat identique",self.n_iter)
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
        predictions = []
        while self.n_iter < nb_iter:
            self.update()
            predictions.append(self.y)
        return predictions

    def train(self,inputs,expected,starting_time):
        X = np.zeros((len(inputs)-starting_time,self.N))
        G = np.zeros((len(inputs)-starting_time,self.number_output))
        for i in range(starting_time):
            self.update(input[i])
        for i in range(starting_time, len(inputs)):
            X[i-starting_time] = np.copy(self.x)
            G[i-starting_time] = expected[i]



        pass


    def save(self,name):
        pass

    @classmethod
    def load(self,name):
        pass


#Mackey glass function import.
file = open("mgdata.dat.txt")
mackey_glass = list(map(lambda x : [float(x.split(" ")[1].split("\n")[0])] ,file.readlines()))
total_len = len(mackey_glass)

def compare_MG(esn,starting_iteration):
    plt.plot([i for i in range(starting_iteration,total_len)], [mackey_glass[i] for i in range(starting_iteration,total_len)],label = "Mackey Glass")
    plt.plot([i for i in range(starting_iteration,total_len)],esn.simulation(total_len-starting_iteration,[mackey_glass[i] for i in range(starting_iteration)]),label = "ESN response")

    plt.legend()
    plt.show()


test= ESN(number_neurons = 100, proba_connexion = 0.2, number_input = 1, number_output = 1, spectral_radius = 0.5)
print(max(abs(np.linalg.eig(test.W)[0]))) #Check wether the spectral radius is respected.


'''
test.update(mackey_glass[0])
test.update(mackey_glass[1])
for i in range(100):
    test.update()
'''
compare_MG(test,100)
