import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

##Basic functions, used in this particular ESN class. Please note that since it is not a general one, they aren't modulable in this case, and changes has to be done by hand.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class ESN:
    '''
    Notes that this is a specific Echo State Network for training purpose, without the maximum features.
    It may ultimately be a basic one for spatialisation purpose.
    '''
    def __init__(self,number_neurons, proba_connexion,number_input,number_output,spectral_radius,leak_rate,noise):
        '''
        Creates an instance of ESN given some parameters
        :parameters:
            - number_neurons : The number of neurons in the reservoir
            - proba_connexion : the probability of connecting to another neuron (or the density of connexion in the reservoir)
            - number_input : How many input neurons.
            - number_output : How many output neurons.
            - spectral_radius : the desired spectral radius, depending on how lastong we want the memory to be.
            - leak_rate: The leak_rate on every update, symbolize the amount of information kept/lost.
        '''
        self.number_input = number_input
        self.number_output = number_output
        self.N = number_neurons  #How many neurons in our reservoir
        self.x = 2*np.random.rand((self.N))-1   #Internal state of the reservoir. Initialisation might change
        self.y = 2*np.random.rand((self.number_output))-1
        self.n_iter = 0
        self.leak_rate = leak_rate
        self.W = np.zeros((self.N,self.N))  #The internal weigth matrix
        #The loop will connect the reservoir with a certain probability.
        for i in range(self.N):
            for j in range(self.N):
                is_connected = (random.random()<proba_connexion)
                if is_connected and i!=j:
                    self.W[i,j] = 2*random.random()-1   #Uniformly between -1 and 1, maybe to change
        eigenvalue = max(abs(np.linalg.eig(self.W)[0]))
        if eigenvalue == 0.0:
            raise Exception("Null Maximum Eigenvalue")
        self.W *= spectral_radius/eigenvalue            #We normalize the weight matrix to get the desired spectral radius.
        self.W_in = (2*np.random.rand(self.N,self.number_input) - 1)    #We initialise between -1 and 1 uniformly, maybe to change
        self.W_out = 2*np.random.rand(self.number_output, self.number_output + self.N + self.number_input) - 1
        self.W_back = 2*np.random.rand(self.N,self.number_output) - 1 #The Feedback matrix
        self.noise = noise




    def update(self,input = [],addNoise = False):
        '''
        Advance the process by 1 step
        '''
        if input == []:
            input = np.zeros((self.number_input))

        test= np.copy(self.x) # TODO: Delete test when it is working properly.
        matrix1 = np.dot(self.W_in, input)
        matrix2 = np.dot(self.W, self.x)
        matrix3 = np.dot(self.W_back, self.y)
        if addNoise:
            self.x = (1-self.leak_rate) * self.x + self.leak_rate * tanh(matrix1 + matrix2 + matrix3 + self.generateNoise())
        else:
            self.x = (1-self.leak_rate) * self.x + self.leak_rate * tanh(matrix1 + matrix2 + matrix3)
        if (np.array_equal(test,self.x)):
            print("Etat identique",self.n_iter)

        tab1 = np.concatenate((input,self.x,self.y))
        self.y = np.dot(self.W_out,tab1)                     #We use a linear output.
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
            predictions.append(self.y)
            self.update()
        return predictions

    def train(self,inputs,expected,starting_time):
        X = np.zeros((len(inputs)-starting_time,self.N+self.number_input+self.number_output))
        G = np.zeros((len(inputs)-starting_time,self.number_output))
        for i in range(starting_time):  #The warmup period
            self.update(inputs[i],addNoise = True)
        for i in range(starting_time, len(inputs)):
            X[i-starting_time] = np.concatenate((inputs[i],self.x,self.y))
            self.update(inputs[i],addNoise = True)
            G[i-starting_time] = expected[i-starting_time]
        for j in range(self.number_output):
            model = LinearRegression()
            print(X.shape,G[:,j].shape)
            model.fit(X,G[:,j])
            newWeights = model.coef_
            print(self.W_out)
            self.W_out[j] = newWeights
            print(self.W_out)

    def generateNoise(self):
        return ((2 * np.random.rand(self.number_output)-1) * self.noise) #A random vector beetween -noise and noise

    def save(self,name):
        pass

    @classmethod
    def load(self,name):
        pass


#Mackey glass function import.
file = open("mgdata.dat.txt")
mackey_glass = list(map(lambda x : [float(x.split(" ")[1].split("\n")[0])] ,file.readlines()))
file.close()
total_len = len(mackey_glass)
mackey_glass = mackey_glass
print(total_len)

def compare_MG(esn,starting_iteration):
    simu = esn.simulation(nb_iter = total_len-starting_iteration, initial_inputs = [mackey_glass[i] for i in range(starting_iteration)])
#    print(simu)
    plt.plot([i for i in range(starting_iteration,total_len)], [mackey_glass[i] for i in range(starting_iteration,total_len)],label = "Mackey Glass")
    plt.plot([i for i in range(starting_iteration,total_len)],simu,label = "ESN response")

    plt.legend()
    plt.show()


test= ESN(number_neurons = 400, proba_connexion = 0.3, number_input = 1, number_output = 1, spectral_radius = 0.7,leak_rate = 0.9, noise = 0.0001)
test.W_back *= 0
print(max(abs(np.linalg.eig(test.W)[0]))) #Check wether the spectral radius is respected.

test.train(mackey_glass,mackey_glass,1000)

'''
test.update(mackey_glass[0])
test.update(mackey_glass[1])
for i in range(100):
    test.update()
'''

compare_MG(test,100)
print(test.x)
