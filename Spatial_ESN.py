import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.spatial.distance as distance
import json
import time
import subprocess

# Default parameters
_data = {
    "seed"           : 1,
    "len_training" : 1000,
    "len_warmup" : 100,
    "epsilon" : 1e-8,
    "number_neurons" : 10**2,
    "sparsity" : 0.4,          #The probability of connection to the input/output (distance-based)
    "intern_sparsity" : 0.1,   #The probability of connection inside of the reservoir, generally lower
    "spectral_radius" : 1.25,
    "leak_rate" : 0.5,
    "display_animation" : True,
    "label_input" : "Mackey Glass",
    "timestamp"      : "",
    "git_branch"     : "",
    "git_hash"       : "",
}




##Basic functions, used in this particular ESN class. Please note that since it is not a general one, they aren't modulable in this case, and changes has to be done by hand.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class Spatial_ESN:
    '''
    Notes that this is a specific Echo State Network for training purpose, without the maximum features.
    It may ultimately be a basic one for spatialisation purpose.
    '''
    def __init__(self,number_neurons, sparsity,intern_sparsity,number_input,number_output,spectral_radius,leak_rate,noise):
        '''
        Creates an instance of ESN given some parameters
        :parameters:
            - number_neurons : The number of neurons in the reservoir
            - sparsity : Used for the connexion in the spatial reservoir. The higher it is, the more connections there are.
            - number_input : How many input neurons.
            - number_output : How many output neurons.
            - spectral_radius : the desired spectral radius, depending on how lastong we want the memory to be.
            - leak_rate: The leak_rate on every update, symbolize the amount of information kept/lost.

        '''
        self.number_input = number_input
        self.number_output = number_output
        self.N = number_neurons  #How many neurons in our reservoir
        self.leak_rate = leak_rate
        self.noise = noise
        self.isRecording = False
        size = int(np.sqrt(self.N))
        if size**2 == self.N:
            self.squared_size = size
        else:
            self.squared_size = -1
        self.sparsity = sparsity
        self.intern_sparsity = intern_sparsity

        self.reset_reservoir(completeReset = True)  #Sets the internal states and weight matrixes.

        #Values initialized later.
        self.len_warmup = -1
        self.len_training = -1

    def reset_reservoir(self,completeReset = False):
        '''
        Resets the values of the internal states. Warmup should be redone after a call to this function.
        completeReset = True will reset all the weights, used for (re)initialization
        '''

        self.x = np.zeros((self.N),dtype = [("activity",float),("position",float,(2,)),("mean",float)])
        self.x["activity"] = np.random.uniform(-1,1,(self.N,))   #Internal state of the reservoir. Initialisation might change

        self.x["mean"] = self.x["activity"]
        self.x["position"][:,0] = np.random.uniform(0,1,(self.N))
        self.x["position"][:,1] = np.random.uniform(0,0.5,(self.N))
        self.istrained = False
        if completeReset:       #Initialization of the weights matrixes.
            self.n_iter = 0

            self.W = 0.5 * np.random.uniform(-1,1,(self.N,self.N))  #The internal weight matrix
            distances = distance.cdist(self.x["position"],self.x["position"]) #Computes the distances between each nodes, used for
        #    self.W *= np.random.uniform(-1,1,self.W.shape) < self.sparsity * (1- np.eye(self.N))
            intern_connections = distances < np.random.uniform(0,self.sparsity,(distances.shape))
            self.W *= intern_connections * (1-np.eye(self.N))   #Connects spatially
            eigenvalue = np.max(np.abs(np.linalg.eigvals(self.W)))
            if eigenvalue == 0.0:
                print(self.W)
                raise Exception("Null Maximum Eigenvalue")
            self.W *= spectral_radius/eigenvalue            #We normalize the weight matrix to get the desired spectral radius.

            self.W_in = np.random.uniform(-1,1,(self.N, 1 + self.number_input))    #We initialise between -1 and 1 uniformly, maybe to change. The added input will be the bias
            #self.W_in = np.ones((self.N, 1 + self.number_input)) #To better visualize, but to delete !
            connexion_in = np.tile(self.x["position"][:,0],(self.number_input + 1,1)).T/(self.intern_sparsity) < (np.random.uniform(0,1,self.W_in.shape))
            self.W_in *= connexion_in

            self.W_out = np.random.uniform(-1,1,(self.N,self.number_output))
            #self.connexion_out = np.tile(1-self.x["position"][:,0],(self.number_output,1)).T < (np.random.uniform(0,self.sparsity,self.W_out.shape))
            self.connexion_out = (1-self.x["position"][:,0]) < (np.random.uniform(0,self.sparsity,self.N))  #The neurons connected are connected to all of the exit neurons. (Makes the training easier)
            if self.number_output == 1:
                self.W_out *= np.tile(self.connexion_out[np.newaxis].T,(self.number_output,1))
            else:
                self.W_out *= np.tile(self.connexion_out,(self.number_output,1))

            self.W_back = np.random.uniform(-1,1,(self.N,self.number_output))  #The Feedback matrix, not used in the test cases.
            self.y = np.zeros((self.number_output))


    def update(self,input = np.array([]),addNoise = False):
        '''
        Advance the process by 1 step, given some input if needed.
        '''
        if input.size == 0:
            input = np.zeros((self.number_input))
        else:
            input = np.array(input)
        u = 1.0 , input
        matrixA = np.dot(self.W_in, u)
        matrixB = np.dot(self.W , self.x["activity"])
        matrixC = 0 #self.W_back @ self.y
        if addNoise:
            self.x["activity"] = (1-self.leak_rate) * self.x["activity"] + self.leak_rate * tanh(matrixA + matrixB + matrixC + self.generateNoise())
        else:
            self.x["activity"] = (1-self.leak_rate) * self.x["activity"] + self.leak_rate * tanh(matrixA + matrixB + matrixC)
        if np.isnan(np.sum(self.x["activity"])):
            raise Exception("Nan in matrix x : {} \n matrix y: {}".format(self.x["activity"],self.y))

        if self.isRecording:
            self.record_state()

        if self.istrained:
            self.y = np.dot(self.W_out,self.x["activity"])                     #We use a linear output.
        else:
            self.y = input
            #print(input-self.y)
        self.n_iter +=1
        self.x["mean"] = (self.x["mean"]*self.n_iter +self.x["activity"])/(self.n_iter + 1)

    def warmup(self,initial_inputs):
        print("---Beginning warmup---")
        for input in initial_inputs:
            self.update(input)  # Warmup period, should have an initialised reservoir at this point.
        print("---Warmup done---")


    def train(self,inputs,expected):
        '''
        Trains the ESN given an input, for all the duration of the input, using linear regression. This should change when spatialized.
        '''
        print("---Beginning training---")
        X = np.zeros((len(inputs),self.N))
        for i in range(1,len(inputs)):
            X[i] = self.x["activity"]*self.connexion_out
            self.update(inputs[i],addNoise = True)
        newWeights = np.dot(np.dot(expected.T,X), np.linalg.inv(np.dot(X.T,X) + epsilon*np.eye(self.N)))
        self.W_out = newWeights
        print("---Training done---")
        self.istrained = True
        self.y =  np.dot(self.W_out,self.x["activity"])   #Output state of the reservoir. It is directly computed from the internal state, and will allow a free-run later

    def generateNoise(self):
        return np.random.uniform(-self.noise,self.noise,(self.number_output)) #A random vector beetween -noise and noise

    def simulation(self,inputs,expected, nb_iter,len_warmup = 0 ,len_training = 0,reset = False):
        '''
        Simulates the behaviour of the ESN given :
        - input : a starting sequence, wich will be followed.
        - nb_iter: number of iteration the ESN will run alone (ie simulate)
        - len_warmup: number of iterations of the warmup sequence.
        - len_training: number of iteration of the training sequence.
        - reset: wether the coeffs of the ESN are reset or not. This will not undo training, and you must use reset_reservoir manually if you want to.
        Input must at least be of length len_warmup + len_training.
        '''
        self.len_warmup = len_warmup
        self.len_training = len_training
        assert len_warmup + len_training <= len(inputs), "Insufficient input size"
        if reset :
            self.reset_reservoir()  #initial reset for multiple calls
        if len_warmup > 0 :
            self.warmup(inputs[:len_warmup])
        if len_training > 0 :
            self.train(inputs[len_warmup:len_warmup+len_training],expected[len_warmup:len_warmup+len_training])
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
        scat = ax.scatter(x = self.x["position"][:,0],y = self.x["position"][:,1], c = self.x["activity"], vmin = -1 , vmax = 1)
        scat.set_sizes(10*np.ones(self.N))
        plt.colorbar(scat)
        def update_frame(i):
            scat.set_array(self.historic[i])
            ax.set_title("{}: Step n°{}".format("Warmup" if i < len_warmup else ("Training" if i < len_warmup + len_training else "Prediction"),i))
        anim = animation.FuncAnimation(fig, update_frame,frames = np.arange(1,len(self.historic)),interval = 25)
        if name != "":
            print("---Saving the animation---")
            anim.save(name+".mp4", fps=30)
            print("---Saving done---")
        if isDisplayed:
            plt.show()

        plt.close()
        self.isRecording = False
        self.historic = []

    def record_state(self):
        '''
        Stores an array of the current activity state.
        '''
        assert self.squared_size !=-1, "Non squared number of neurons: {}".format(self.N)
        self.historic.append(np.copy(self.x["activity"]))

    def disp_connectivity(self, i = -1):
        '''
        Displays the connections inside the reservoir, majoritarly to see what happens during spatialization. If i is given, it will simply display the connection from i to others neurons
        '''
        connexion_in = (self.W_in != 0)
        if len(self.connexion_out.shape) == 1:
            connexion_out = self.connexion_out[np.newaxis].T
        else:
            connexion_out = self.connexion_out
        intern_connections = (self.W != 0)
        figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,20))

        print("Number of connection to the reservoir : ",np.sum(connexion_in))
        print("Number of connection inside the reservoir : ",np.sum(intern_connections))
        print("Number of connection to the output : ",np.sum(connexion_out))

        figure.suptitle("{} neurons, sparsity = {} ".format(self.N, self.sparsity))
        axes[1].scatter(self.x["position"][:,0], self.x["position"][:,1], c = [1 if connexion_out[i,:].any() else 0 for i in range(self.N)])
        axes[0].scatter(self.x["position"][:,0], self.x["position"][:,1], c = [1 if connexion_in[i,:].any() else 0 for i in range(self.N)])
        axes[0].set_title("Neurons connected to the input")
        axes[1].set_title("Neurons connected to the output")

        axes[2].scatter(self.x["position"][:,0], self.x["position"][:,1])
        axes[2].set_aspect(1)
        axes[1].set_aspect(1)
        axes[0].set_aspect(1)
        if i == -1:
            for i in range(self.N):
                for j in range(self.N):
                    if self.W[i,j] != 0:
                        l = axes[2].arrow(self.x["position"][i,0],self.x["position"][i,1],(self.x["position"][j,0] - self.x["position"][i,0]), (self.x["position"][j,1]- self.x["position"][i,1]),lw = 0.1)
                        #axes[2].plot(self.x["position"][i,0],self.x["position"][i,1], self.x["position"][j,0], self.x["position"][j,1])
        else:
            for j in range(self.N):
                if self.W[i,j] != 0:
                    l = axes[2].arrow(self.x["position"][i,0],self.x["position"][i,1],(self.x["position"][j,0] - self.x["position"][i,0]), (self.x["position"][j,1]- self.x["position"][i,1]),lw = 0.1,color = 'Red')
        axes[2].set_title("Intern connection "+("for a single neuron" if i ==-1 else "for all neurons"))
        plt.show()
        plt.close()


def compare_result(esn,input,label_input ,len_warmup,len_training,nb_iter = -1, displayAnim = True, savename = ""):
    '''
    Trains the network, and display both the expected result and the network output. Can also save/display the plot of the inner working.
    Inputs:
        - esn : an instance of Spatial_ESN
        - input : the input series
        - label_input : the name for the plot
        - len_warmup: For how long the ESN is warmupped
        - len_training : the length of the training
        - nb_iter : for how long the simulation is done after training. Computed by default to fit the length of input
        - displayAnim : Wether the internal state is plotted
        - savename: optionnal, where the .mp4 is generated. If not filled, it won't be generated.
    '''
    display = displayAnim or (savename != "")
    #esn.disp_connectivity()
    if display:
        esn.begin_record()
    if nb_iter ==-1:
        nb_iter = len(input) - len_warmup - len_training
    simu = esn.simulation(nb_iter = nb_iter, inputs = input,expected = input, len_warmup = len_warmup, len_training = len_training, reset = False )
    if display:
        esn.end_record(savename,isDisplayed = displayAnim)
    esn.disp_connectivity(i= -1)
    plt.plot(range(len_warmup+len_training,nb_iter+len_warmup+len_training), input[len_warmup+len_training:nb_iter+len_warmup+len_training],label = label_input)
    plt.plot(range(len_warmup+len_training,nb_iter+len_warmup+len_training), simu, label = "ESN response")
    plt.title("ESN with {} neurons, sparsity = {}".format(esn.N,esn.sparsity))
    plt.legend()
    plt.show()




#File and json Handling

def get_git_revision_hash():
    """ Get current git hash """
    answer = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    return answer.decode("utf8").strip("\n")

def get_git_revision_branch():
    """ Get current git branch """
    answer = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    return answer.decode("utf8").strip("\n")

def default():
    """ Get default parameters """
    _data["timestamp"] = time.ctime()
    _data["git_branch"] = get_git_revision_branch()
    _data["git_hash"] = get_git_revision_hash()
    return _data

def save(filename, data=None):
    """ Save parameters into a json file """
    if data is None:
       data = { name : eval(name) for name in _data.keys()
                if name not in ["timestamp", "git_branch", "git_hash"] }
    data["timestamp"] = time.ctime()
    data["git_branch"] = get_git_revision_branch()
    data["git_hash"] = get_git_revision_hash()
    with open(filename, "w") as outfile:
        json.dump(data, outfile)

def load(filename):
    """ Load parameters from a json file """
    with open(filename) as infile:
        data = json.load(infile)
    return data

def dump(data):
    for key, value in data.items():
        print(f"{key:15s} : {value}")


if __name__  == "__main__":
    save("temp.txt", _data)
    data = load("temp.txt")
    dump(data)
    locals().update(data)
    save("temp.txt")

    np.random.seed(seed)

    #Mackey glass function import.
    if label_input == "Mackey Glass":
        input = np.load("mackey-glass.npy")[np.newaxis].T
    elif label_input == "Sinus":
        input = np.sin(np.arange(start = 0,stop = 1000,step = 1/10))
    #Creating the ESN
    test= Spatial_ESN(number_neurons = number_neurons, sparsity = sparsity,intern_sparsity = intern_sparsity, number_input = 1, number_output = 1, spectral_radius = spectral_radius, leak_rate = leak_rate, noise = 0)
    test.W_back *= 0
    print("Effective spectral radius :",max(abs(np.linalg.eig(test.W)[0]))) #Check wether the spectral radius is respected.
    compare_result(test,input = input,len_warmup = len_warmup,len_training = len_training, nb_iter = 2000,displayAnim = display_animation ,savename = "",label_input = label_input + " series")
