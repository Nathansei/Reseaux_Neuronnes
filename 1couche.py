import numpy as np

import numpy as np  # linear algebra
import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt

#
# MNIST Data Loader Class
#
class MnistDataloader(object):

    def __init__(self): # training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath):

        #
        # Set file paths based on added MNIST Datasets
        #
        input_path = 'Datas'
        training_images_filepath = input_path + '/train-images.idx3-ubyte'
        training_labels_filepath = input_path + '/train-labels.idx1-ubyte'
        test_images_filepath = input_path + '/t10k-images.idx3-ubyte'
        test_labels_filepath = input_path + '/t10k-labels.idx1-ubyte'


        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

#
# Verify Reading Dataset via MnistDataloader class
#






#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15)
        index += 1

#
# Load MINST dataset
#



# if __name__=="__main__":
if True:

    mnist_dataloader = MnistDataloader()
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    #
    # Show some random training and test images
    #
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, 60000)
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

    for i in range(0, 5):
        r = random.randint(1, 10000)
        images_2_show.append(x_test[r])
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

    show_images(images_2_show, titles_2_show)
class NeuronneFactice:

    def __init__(self, neuronnes, app):
        self.neuronnes = neuronnes
        self.app = app
        self.poids = []
        self.biais = []
        for i in range(len(neuronnes) - 1):
            n = neuronnes[i]
            n_plus_un = neuronnes[i + 1]
            W = np.random.uniform(-np.sqrt(6 / n), np.sqrt(6 / n), size=(n, n_plus_un))
            self.poids.append(W)
            B = np.zeros((1, n_plus_un))
            self.biais.append(B)

    def feedforward(self,inputs):
        activation = inputs
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

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def cout(self, liste_res, liste_res_attendu):
        cout=0
        for i in range(len(liste_res)):
            cout += abs(liste_res_attendu[i] - liste_res[i])
        return cout

    def backwardpropagation(self,delta,resultat):
        w=self.poids
        for n in range(len(self.poids)):
            for i in range(len(self.poids[n])):
                for j in range(len(self.poids[n][i])):
                    w[n][i][j] = w[n][i][j] + self.app * resultat[n][i] * delta[n][j]

    def backwardpropagation2(self,delta,resultat):
        for n in range(len(self.poids)):
            gradient = np.dot(resultat[n].T,delta[n+1])
            self.poids[n]+=self.app*gradient
            self.biais[n] += self.app * delta[n + 1]
    def delta(self,label,resultats):
        delta = [np.zeros_like(r) for r in resultats]
        w = self.poids
        delta[-1] = (label - resultats[-1])
        for i in range(len(self.poids)-2,-1,-1):
            for j in range(len(self.biais[i])):
                mat=np.dot(delta[i+1],w[i].T)
                somme=0
                for ligne in range(len(mat)):
                    somme += mat[ligne]
                delta[i][j]=resultats[i][j]*(1-resultats[i][j])*somme
        return delta

    def delta_vectorise(self, label_vector,resultats):
        deltas = [np.zeros_like(r) for r in resultats]
        output = resultats[-1]
        error = (label_vector - output)
        deltas[-1] = error * self.sigmoid_deriv(output)

        for i in range(len(self.poids) - 1, -1, -1):
            error_prop = np.dot(deltas[i + 1], self.poids[i].T)

            # Dérivée locale
            d_activ = self.sigmoid_deriv(resultats[i])

            deltas[i] = error_prop * d_activ

        return deltas
    def softmax(self,liste):
        somme = 0
        for i in range(len(liste)):
            somme += liste[i]
        for i in range(len(liste)):
            liste[i] = liste[i]/somme



def input_Mnist_trad():
    dataloader = MnistDataloader()
    (x_train, label_train), (x_test, label_test) = dataloader.load_data()
    xtrain = np.array(x_train)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtrain = xtrain / 255.0
    xtest = np.array(x_test)
    xtest = xtest.reshape(xtest.shape[0], -1)
    xtest = xtest / 255.0
    return (xtrain, label_train), (xtest, label_test)

(x_train, label_train), (x_test, label_test)=input_Mnist_trad()
def one_hot(label, classes=10):
    vecteur = np.zeros((1, classes))
    vecteur[0][label] = 1
    return vecteur

def trainingtesting(x_train,label_train,x_test, label_test,boucle=200):

    perc = NeuronneFactice([784, 128, 64, 10], 0.15)
    #training
    for j in range(boucle):
        print(j)

        for i in range(len(x_train)):
            pixel=x_train[i].reshape(1, -1)
            label=one_hot(label_train[i])
            resultats = perc.feedforward(pixel)
            delta = perc.delta_vectorise(label, resultats)
            perc.backwardpropagation2(delta, resultats)
    #test
    somme = 0
    for i in range(len(x_test)):
        pixel=x_test[i].reshape(1, -1)

        resultats = perc.feedforward(pixel)
        reponse=np.argmax(resultats[-1])

        if reponse==label_test[i]:
            somme+=1
    print(somme)
    print(len(x_test))
    print("L'IA reconnait suite a son entrainement :",(somme/len(label_test))*100,"% des images")

trainingtesting(x_train, label_train,x_test, label_test)

