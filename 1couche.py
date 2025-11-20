import numpy as np


class NeuronneFactice:

    def __init__(self,neuronnes):
        self.neuronnes = neuronnes # [4,3,2] par exemple
        self.poids = [[[ np.random.uniform(-np.sqrt(6/self.neuronnes[i]),np.sqrt(6/self.neuronnes[i])) for _ in range(neuronnes[i+1])] for _ in range(neuronnes[i])] for i in range(len(neuronnes)-1)]
        self.biais = [[1 for _ in range(neuronnes[i])] for i in range(1,len(neuronnes)-1)]



    def feedforward(self, nombres, poids, biais, fonction):
        x = 0
        for i in range(len(nombres)):
            x += nombres[i] * poids[i]
        if fonction == 'tanh':
            return self.tanh(x + biais)
        if fonction == 'sigmoid':
            return self.sigmoid(x + biais)
        if fonction == 'relu':
            return self.relu(x + biais)

    def relu(self, x):
        return max(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def cout(self, liste_res, liste_res_attendu):
        cout=0
        for i in range(len(liste_res)):
            cout += abs(liste_res_attendu[i] - liste_res[i])
        return cout

    def backwardpropagation(self):
    
        pass

    def softmax(self,liste):
        somme = 0
        for i in range(len(liste)):
            somme += liste[i]
        for i in range(len(liste)):
            liste[i] = liste[i]/somme


class Neuronne():
    def __init__(self,nb_entree,nb_neuronnes):
        self.nb_entree=nb_entree
        self.nb_neuronnes=nb_neuronnes
        self.poids=[[(np.random.uniform(-np.sqrt(6/self.nb_entree),np.sqrt(6/self.nb_entree))) for _ in range(self.nb)]]

