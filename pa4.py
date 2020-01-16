#using BP neural network to be the classfier

import csv
import math
import random
#import string
import matplotlib.pyplot as plt
import pickle

random.seed(0)

# retun a random number between x and y-x
def rand(x, y):
    return (y-x)*random.random() + x

# Define tanh activation function as the hidden layer of neurons
def tanh(x):
    return math.tanh(x)

# Define sigmoid activation function as the output layer of neurons
def sigmoid(x):
    return 1/(1+math.exp(-x))

class Unit:
    def __init__(self, length):
        self.weight = [rand(-0.2, 0.2) for i in range(length)]
        self.change = [0.0] * length
        self.threshold = rand(-0.2, 0.2)
   
    def calc(self, sample):
        self.sample = sample[:]
        partsum = sum([i * j for i, j in zip(self.sample, self.weight)]) - self.threshold
        self.output = tanh(partsum)
        return self.output
    def update(self, diff, rate=0.5, factor=0.1):
        change = [rate * x * diff + factor * c for x, c in zip(self.sample, self.change)]
        self.weight = [w + c for w, c in zip(self.weight, change)]
        self.change = [x * diff for x in self.sample]
    def get_weight(self):
        return self.weight[:]
    def set_weight(self, weight):
        self.weight = weight[:]


class Layer:
    def __init__(self, input_length, output_length):
        self.units = [Unit(input_length) for i in range(output_length)]
        self.output = [0.0] * output_length
        self.ilen = input_length
    def calc(self, sample):
        self.output = [unit.calc(sample) for unit in self.units]
        return self.output[:]
    def update(self, diffs, rate=0.5, factor=0.1):
        for diff, unit in zip(diffs, self.units):
            unit.update(diff, rate, factor)
    def get_error(self, deltas):
        def _error(deltas, j):
            return sum([delta * unit.weight[j] for delta, unit in zip(deltas, self.units)])
        return [_error(deltas, j) for j  in range(self.ilen)]
    def get_weights(self):
        weights = {}
        for key, unit in enumerate(self.units):
            weights[key] = unit.get_weight()
        return weights
    def set_weights(self, weights):
        for key, unit in enumerate(self.units):
            unit.set_weight(weights[key])



class BPNNet:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no
        self.hlayer = Layer(self.ni, self.nh)
        self.olayer = Layer(self.nh, self.no)
        self.Error = []

    def calc(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.ai = inputs[:] + [1.0]

        # hidden activations
        self.ah = self.hlayer.calc(self.ai)
        # output activations
        self.ao = self.olayer.calc(self.ah)


        return self.ao[:]


    def update(self, targets, rate, factor):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [sigmoid(ao) * (target - ao) for target, ao in zip(targets, self.ao)]

        # calculate error terms for hidden
        hidden_deltas = [sigmoid(ah) * error for ah, error in zip(self.ah, self.olayer.get_error(output_deltas))]

        # update output weights
        self.olayer.update(output_deltas, rate, factor)

        # update input weights
        self.hlayer.update(hidden_deltas, rate, factor)
        # calculate error
        return sum([0.5 * (t-o)**2 for t, o in zip(targets, self.ao)])


    def test(self, patterns):
        write_data=[('ID','IsSinkhole')]
        index = 0
        for p in patterns:
            
            write_data.append((index,self.calc(p[0])[0]))
            index=index+1
        with open('pred.csv', 'w', newline='') as t_file:

           csv_writer = csv.writer(t_file)

           for l in write_data:

               csv_writer.writerow(l)

    def train(self, patterns, iterations=10000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.calc(inputs)
                error = error + self.update(targets, N, M)
            if i % 100 == 0:
                print('...')
            self.Error.append(error)
    def save_weights(self, fn):
        weights = {
                "olayer":self.olayer.get_weights(),
                "hlayer":self.hlayer.get_weights()
                }
        with open(fn, "wb") as f:
            pickle.dump(weights, f)
    def load_weights(self, fn):
            with open(fn, "rb") as f:
                weights = pickle.load(f)
                self.olayer.set_weights(weights["olayer"])
                self.hlayer.set_weights(weights["hlayer"])
                
    def get_Error(self):
        return self.Error

def readData(filename):
    raw_data = []
    #load data from *.csv file
    with open(filename) as csvfile:
        reader=csv.reader(csvfile)
        for item in reader:
                raw_data.append(item)

    n = len(raw_data)
    m = len(raw_data[0]) 

    #delete the fist line
    del raw_data[0]

   
   # delete the first column
    
    for item in range(n-1):
            del raw_data[item][0]

    n = len(raw_data)
    m = len(raw_data[0])

    
    #Turn 'str' into 'float'
   
    for i in range(n):
            for j in range(m):
                 raw_data[i][j]= float(raw_data[i][j])             

    for i in range(n):
            raw_data[i][0]=raw_data[i][0:m-1]  #store data
            raw_data[i][1]=raw_data[i][m-1:]   #store label
            del raw_data[i][2:m]
    #Data normalization
    for i in range(n):
     for j in range(m-1):
         raw_data[i][0][j]=raw_data[i][0][j]/sum(raw_data[i][0])

    return raw_data, m-1


def readTestData(filename):
    raw_data = []
    #load data from *.csv file
    with open(filename) as csvfile:
        reader=csv.reader(csvfile)
        for item in reader:
                raw_data.append(item)
  
    n = len(raw_data)
    m = len(raw_data[0]) 

    #delete the fist line
    
    del raw_data[0]

   
    #delete the fist column
    
    for item in range(n-1):
            del raw_data[item][0]

    n = len(raw_data)
    m = len(raw_data[0])

    
    #Turn 'str' into 'float'
   
    for i in range(n):
            for j in range(m):
                 raw_data[i][j]= float(raw_data[i][j])             

    for i in range(n):
            raw_data[i][0]=raw_data[i][0:m]  #store date， no label        
            del raw_data[i][1:m]
    #Data normalization
    for i in range(n):
     for j in range(m):
         raw_data[i][0][j]=raw_data[i][0][j]/sum(raw_data[i][0])

    return raw_data







   
        
def KarstSinkhole():
   
    train_data,num = readData('pa4_train.csv')
	# create a BPnn，num-input， 10-hidden ，1-output
    net = BPNNet(num, 10, 1)
    # start strainng
    print('training ...')

    #iterations，N-learning rate，M-momentum factor
    net.train(train_data,iterations=1000, N=0.01, M=0.1)
    #Get errors during training
    performance = net.get_Error()
           
    # test it

    test_data= readTestData('pa4_test.csv') 
    net.save_weights("KarstSinkhole.weights")
    
    net.test(test_data)

    
    #show errors
    x=[]
    for i in range(len(performance)):
        x.append(i)
        
    plt.plot(x,performance)
    plt.show()


if __name__ == '__main__':
    KarstSinkhole()
