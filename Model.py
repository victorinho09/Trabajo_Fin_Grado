from keras import Sequential, Input
from keras.src.layers import Dense
from numpy.ma.core import shape

from funciones import create_model


class  Model():
    def __init__(self,X_train,y_train):

        self.model = Sequential()

        #Se inicializan ciertas variables con valores por defecto, ya que no hay más referencias en la primera instancia

        self.num_dense_layers = 2

        #Array: 1º valor -> 1º capa ...
        self.num_neurons_per_dense_layer = [64,32]

        #Depende de la estructura de y_train
        self.num_neurons_output_layer = y_train.shape[1]

        #Depende de la estructura de X_train
        self.num_neurons_input_layer = (X_train.shape[1],)

        #De momento, se usa solo el optimizer
        self.learning_rate = None
        self.optimizer = 'adam'

        self.loss = 'categorical_crossentropy'  # uso categorical_crossentropy cuando las etiquetas están codificadas con one-hot encoder. Si no usaría: sparse_categ_cross

        #Array: 1º valor -> 1 capa oculta ...
        self.dense_layers_activation_function = ['relu','relu']
        self.output_layer_activation_function = 'softmax'
        self.metrics = []

        self.create_model()


    def create_model(self):

        self.model.add(Input(self.num_neurons_input_layer))

        #Se añaden el resto de capas del modelo
        for num_neurons_layer,i in range(self.num_neurons_per_dense_layer):
            self.model.add(Dense(num_neurons_layer, activation= self.dense_layers_activation_function[i]))

        #Se añade capa de salida. La función de activación corresponde al último
        self.model.add(Dense(self.num_neurons_output_layer, activation= self.output_layer_activation_function ))

        # Compiling the model. No hace falta especificar las metricas, ya que por defecto, el objeto history que devuelve el metodo fit las contiene
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

