import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

len_training = 1000
len_warmup = 100
epsilon = 1e-8
number_neurons = 30*30
sparsity = 0.6
spectral_radius = 1.25

np.random.seed(1)
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
    def __init__(self,number_neurons, sparsity,number_input,number_output,spectral_radius,leak_rate,noise):
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
        self.reset_reservoir()

        self.leak_rate = leak_rate
        self.W = 0.5 * np.random.uniform(-1,1,(self.N,self.N))  #The internal weigth matrix
        self.W *= np.random.uniform(0,1,self.W.shape)<sparsity
        eigenvalue = np.max(np.abs(np.linalg.eigvals(self.W)))
        if eigenvalue == 0.0:
            print(self.W)
            raise Exception("Null Maximum Eigenvalue")
        self.W *= spectral_radius/eigenvalue            #We normalize the weight matrix to get the desired spectral radius.
        self.W_in = 0.5 * np.random.uniform(-1,1,(self.N, 1 + self.number_input))    #We initialise between -1 and 1 uniformly, maybe to change
        self.W_out = 0.5 * np.random.uniform(-1,1,(self.number_output, (self.N)))
        self.W_back = 0.5 * np.random.uniform(-1,1,(self.N,self.number_output))  #The Feedback matrix
        self.noise = noise
        self.isRecording = False

        #Values initialized later.
        self.squared_size = -1
        self.len_warmup = -1
        self.len_training = -1

    def reset_reservoir(self):
        '''
        Resets the values of the internal states. Warmup should be redone after a call to this function.
        '''
        self.n_iter = 0
        self.x = np.random.uniform(-1,1,(self.N))   #Internal state of the reservoir. Initialisation might change
        self.istrained = False

    def update(self,input = np.array([]),addNoise = False):
        '''
        Advance the process by 1 step, given some input if needed.
        '''
        if input.size == 0:
            input = np.zeros((self.number_input))
        else:
            input = np.array(input)
        #test = np.copy(self.x) # TODO: Delete test when it is working properly.
        u = 1.0 , input
        matrixA = np.dot(self.W_in, u)
        matrixB = np.dot(self.W , self.x)
        matrixC = 0 #self.W_back @ self.y
        if addNoise:
            self.x = (1-self.leak_rate) * self.x + self.leak_rate * tanh(matrixA + matrixB + matrixC + self.generateNoise())
        else:
            self.x = (1-self.leak_rate) * self.x + self.leak_rate * tanh(matrixA + matrixB + matrixC)
        if np.isnan(np.sum(self.x)):
            raise Exception("Nan in matrix x : {} \n matrix y: {}".format(self.x,self.y))

        if self.isRecording:
            self.record_state()

        if self.istrained:
            self.y = np.dot(self.W_out,self.x)                     #We use a linear output.
        else:
            self.y = input
            #print(input-self.y)
        self.n_iter +=1

    def warmup(self,initial_inputs):
        print("---Beginning warmup---")
        for input in initial_inputs:
            self.update(input)  # Warmup period, should have an initialised reservoir at this point.
        print("---Warmup done---")



    def train(self,inputs,expected):
        print("---Beginning training---")
        X = np.zeros((len(inputs),self.N))
        for i in range(1,len(inputs)):
            X[i] = self.x
            self.update(inputs[i],addNoise = True)
        newWeights = np.dot(np.dot(expected.T,X), np.linalg.inv(np.dot(X.T,X) + epsilon*np.eye(self.N)))
        self.W_out = newWeights
        print("---Training done---")
        self.istrained = True
        self.y = np.zeros((self.number_output))   #Output state of the reservoir. Initialisation might change

    def generateNoise(self):
        return np.random.uniform(-self.noise,self.noise,(self.number_output)) #A random vector beetween -noise and noise


    def simulation(self,inputs,expected, nb_iter,len_warmup,len_training,reset = True):
        '''
        Simulates the behaviour of the ESN given a starting sequence (inputs) for nb_iter iterations. Do the training and warmup if reset is True.
        '''
        self.len_warmup = len_warmup
        self.len_training = len_training
        if reset:
            self.reset_reservoir()  #initial reset for multiple calls
            self.warmup(inputs[:len_warmup])
            #self.train(inputs[len_warmup:len_warmup+len_training],expected[len_warmup:len_warmup+len_training])
            self.train(inputs[len_warmup:len_warmup+len_training],expected[len_warmup -1 :len_warmup -1 +len_training])

        print("---Begining simulation without input---")
        predictions = []
        for _ in range(nb_iter):
            self.update(self.y)
            predictions.append(self.y)
        print("---Simulation done---")
        return predictions

    def begin_record(self):
        self.isRecording = True
        self.historic = []
        self.record_state()

    def end_record(self,name,isDisplayed = False):
        fig = plt.figure()
        ax = plt.subplot(1,1,1, aspect=1, frameon=False)
        title = ax.set_title("Warmup: Step n°0")
        image = ax.imshow(self.historic[0],cmap = 'gray',vmin = -1, vmax = 1)
        def update_frame(i):
            image.set_data(self.historic[i])
            ax.set_title("{}: Step n°{}".format("Warmup" if i< len_warmup else ("Training" if i < len_warmup+len_training else "Prediction"),i))

        anim = animation.FuncAnimation(fig, update_frame,frames = np.arange(1,len(self.historic)),interval = 25)
        if name != "":
            print("---Saving the animation---")
            anim.save(name+".mp4", fps=500)
            print("---Saving done---")
        if isDisplayed:
            plt.show()

        plt.close()
        self.isRecording = False
        self.historic = []

    def record_state(self):
        '''
        Returns a np array of the activity of the internal states. It should be consistent (ie amongst iterations)
        '''
        if self.squared_size ==-1:
            size = int(np.sqrt(self.N))
            assert  size**2 == self.N, "Non squared number of neurons: {}".format(self.N)
            self.squared_size = size
        to_display = self.x.reshape(self.squared_size,self.squared_size)
        self.historic.append(to_display)

#Mackey glass function import.
'''
file = open("mgdata.dat.txt")
mackey_glass = list(map(lambda x : [float(x.split(" ")[1].split("\n")[0])] ,file.readlines()))
file.close()
'''
mackey_glass = np.load("mackey-glass.npy")[np.newaxis].T
total_len = len(mackey_glass)


def compare_MG(esn,nb_iter = -1,display = True, savename = ""):
    if display:
        esn.begin_record()
    if nb_iter ==-1:
        nb_iter = len(mackey_glass) - len_warmup - len_training

    simu = esn.simulation(nb_iter = nb_iter, inputs = mackey_glass,expected = mackey_glass, len_warmup = len_warmup, len_training = len_training, reset = True )
    if display:
        esn.end_record(savename,isDisplayed = True)
    plt.plot(range(len_warmup+len_training,nb_iter+len_warmup+len_training),mackey_glass[len_warmup+len_training:nb_iter+len_warmup+len_training],label = "Mackey_glass series")
    plt.plot(range(len_warmup+len_training,nb_iter+len_warmup+len_training),simu,label = "ESN response")

    plt.legend()
    plt.show()

#Creating the ESN
test= ESN(number_neurons = number_neurons, sparsity = sparsity, number_input = 1, number_output = 1, spectral_radius = spectral_radius, leak_rate = 0.5, noise = 0)
test.W_back *= 0

print(max(abs(np.linalg.eig(test.W)[0]))) #Check wether the spectral radius is respected.

compare_MG(test,nb_iter = 2000,display = True,savename = "")
