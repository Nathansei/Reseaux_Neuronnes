import numpy as np


class NeuronneFactice:

    def __init__(self, neuronnes, app, inputs):
        self.neuronnes = neuronnes
        self.app = app
        self.input = np.array(inputs).reshape(1, -1)
        self.poids = []
        self.biais = []
        for i in range(len(neuronnes) - 1):
            n = neuronnes[i]
            n_plus_un = neuronnes[i + 1]
            W = np.random.uniform(-np.sqrt(6 / n), np.sqrt(6 / n), size=(n, n_plus_un))
            self.poids.append(W)
            B = np.zeros((1, n_plus_un))
            self.biais.append(B)

    def feedforward(self):
        activation = self.input
        resultat = [activation]
        for i in range(len(self.poids)):
            w = self.poids[i]
            b = self.biais[i]
            z = np.dot(activation, w) + b
            activation = self.sigmoid(z)
            resultat.append(activation)
        return resultat


    def feedforwardneur(self, nombres, poids, biais, fonction):
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
        #w[n][i][j] = w[n][i][j] + self.app * resultat[n][i] * delta[n][j]

        pass

    def softmax(self,liste):
        somme = 0
        for i in range(len(liste)):
            somme += liste[i]
        for i in range(len(liste)):
            liste[i] = liste[i]/somme


inputs = [0.5, -1.2]
perc = NeuronneFactice([2, 3, 1], None, inputs)
resultats = perc.feedforward()
