import numpy as np
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from activation import linear, d_linear, relu, d_relu, sigmoid, d_sigmoid, tanh, d_tanh
from tqdm import tqdm  

class FFNN:
    def __init__(self, layers, activations, loss_func, loss_grad, init_method, init_params):
        """
        Parameter:
            layers      : List jumlah neuron tiap layer (misal: [784, 128, 10])
            activations : List fungsi aktivasi untuk tiap layer (kecuali input layer) (misal: [sigmoid, sigmoid])
            loss_func   : Fungsi loss (misal: binary_cross_entropy_loss)
            loss_grad   : Turunan fungsi loss (misal: d_binary_cross_entropy_loss)
            init_method : Fungsi inisialisasi bobot (misal: init_weights_uniform)
            init_params : Parameter untuk inisialisasi bobot dalam bentuk dictionary (misal: {'lower_bound': -1, 'upper_bound': 1, 'seed': 42})
        """
        # buat init method bobot juga belom tau bisa beda-beda apa ga

        self.num_layers = len(layers)
        self.layers = layers
        self.activations = activations
        self.loss_func = loss_func
        self.loss_grad = loss_grad
        self.init_method = init_method
        self.init_params = init_params

        self.weights = []
        self.biases = []
        self.gradients = {}
        for i in range(1, self.num_layers):
            weight_shape = (layers[i-1], layers[i])
            bias_shape = (1, layers[i])
            self.weights.append(self.init_method(weight_shape, **init_params))
            self.biases.append(self.init_method(bias_shape, **init_params))
            self.gradients[f"W{i}"] = np.zeros(weight_shape)
            self.gradients[f"b{i}"] = np.zeros(bias_shape)

    def forward(self, X):
        self.z_values = []
        self.a_values = [X] 
        a = X
        for i in range(self.num_layers - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.activations[i](z)
            self.a_values.append(a)
        return a

    def backward(self, X, y):
        m = X.shape[0]
        y_pred = self.a_values[-1]
        delta = self.loss_grad(y, y_pred)
        
        for i in reversed(range(self.num_layers - 1)):
            z = self.z_values[i]
            act_func = self.activations[i]
            if act_func == sigmoid:
                delta *= d_sigmoid(z)
            elif act_func == relu:
                delta *= d_relu(z)
            elif act_func == tanh:
                delta *= d_tanh(z)
            elif act_func == linear:
                delta *= d_linear(z)
            
            # buat softmax belom tau mo gimana ges

            a_prev = self.a_values[i]
            self.gradients[f"W{i+1}"] = np.dot(a_prev.T, delta) / m
            self.gradients[f"b{i+1}"] = np.sum(delta, axis=0, keepdims=True) / m

            if i != 0:
                delta = np.dot(delta, self.weights[i].T)

    def update_weights(self, learning_rate):
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * self.gradients[f"W{i+1}"]
            self.biases[i] -= learning_rate * self.gradients[f"b{i+1}"]

    def train(self, X_train, y_train, X_val, y_val, batch_size, epochs, learning_rate, verbose):
        """
        Parameter:
            X_train, y_train : Data training
            X_val, y_val     : Data validasi
            batch_size       : Ukuran batch
            epochs           : Jumlah epoch
            learning_rate    : Learning rate untuk update bobot
            verbose          : 0 / 1 untuk menampilkan progress
        """

        history = {"train_loss": [], "val_loss": []}
        num_batches = int(np.ceil(X_train.shape[0] / batch_size))
        
        # harus di cek lagi perlu shuffle apa ga
        for epoch in range(epochs):
            epoch_loss = 0
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            if verbose:
                batch_iter = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}")
            else:
                batch_iter = range(num_batches)
            
            for batch in batch_iter:
                start = batch * batch_size
                end = start + batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                y_pred = self.forward(X_batch)
                loss = self.loss_func(y_batch, y_pred)
                epoch_loss += loss

                self.backward(X_batch, y_batch)
                self.update_weights(learning_rate)
            
            avg_train_loss = epoch_loss / num_batches
            history["train_loss"].append(avg_train_loss)
            
            y_val_pred = self.forward(X_val)
            val_loss = self.loss_func(y_val, y_val_pred)
            history["val_loss"].append(val_loss)
            
            if verbose:
                tqdm.write(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        return history

    def display_model_graph(self):
        Graph = nx.DiGraph()
        for layer in range(self.num_layers):
            for neuron in range(self.layers[layer]):
                Graph.add_node(f"L{layer}_N{neuron}", layer=layer)

        for i in range(1, self.num_layers):
            for j in range(self.layers[i-1]):
                for k in range(self.layers[i]):
                    weight = self.weights[i-1][j, k]
                    grad = self.gradients[f"W{i}"][j, k]
                    Graph.add_edge(f"L{i-1}_N{j}", f"L{i}_N{k}", weight=weight, grad=grad)
        
        pos = {}
        for layer in range(self.num_layers):
            x = layer
            y_values = list(range(self.layers[layer]))
            for idx, neuron in enumerate(range(self.layers[layer])):
                pos[f"L{layer}_N{neuron}"] = (x, -idx)
        
        plt.figure(figsize=(12, 8))
        nx.draw(Graph, pos, with_labels=True, node_color="lightblue", arrows=True)
        plt.title("Struktur Jaringan, Bobot & Gradien")
        plt.show()

    def plot_weight_and_gradient_distribution(self, layers_to_plot):
        # misal = model.plot_weight_and_gradient_distribution[1, 2] (plot untuk layer 1 dan 2)
        n_layers = len(layers_to_plot)
        fig, axes = plt.subplots(nrows=n_layers, ncols=2, figsize=(12, 4 * n_layers))
        
        if n_layers == 1:
            axes = np.array([axes])
        
        for idx, layer in enumerate(layers_to_plot):
            if 1 <= layer < self.num_layers:
                weights = self.weights[layer-1].flatten()
                axes[idx, 0].hist(weights, bins=30, edgecolor='black')
                axes[idx, 0].set_title(f"Distribusi Bobot untuk Layer {layer}")
                axes[idx, 0].set_xlabel("Nilai Bobot")
                axes[idx, 0].set_ylabel("Frekuensi")
                
                grads = self.gradients[f"W{layer}"].flatten()
                axes[idx, 1].hist(grads, bins=30, edgecolor='black')
                axes[idx, 1].set_title(f"Distribusi Gradien untuk Layer {layer}")
                axes[idx, 1].set_xlabel("Nilai Gradien")
                axes[idx, 1].set_ylabel("Frekuensi")
            else:
                axes[idx, 0].text(0.5, 0.5, f"Layer {layer} tidak valid", 
                                horizontalalignment='center', verticalalignment='center')
                axes[idx, 1].text(0.5, 0.5, f"Layer {layer} tidak valid", 
                                horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        plt.show()

    ## ini kalo plotnya mau pagenya dipisah ya 
    # def plot_weight_distribution(self, layers_to_plot):
    #     for i in layers_to_plot:
    #         if 1 <= i < self.num_layers:
    #             weights = self.weights[i-1].flatten()
    #             plt.figure()
    #             plt.hist(weights, bins=30, edgecolor='black')
    #             plt.title(f"Distribusi Bobot untuk Layer {i}")
    #             plt.xlabel("Nilai Bobot")
    #             plt.ylabel("Frekuensi")
    #             plt.show()
    #         else:
    #             print(f"Layer {i} tidak valid untuk distribusi bobot.")

    # def plot_gradient_distribution(self, layers_to_plot):
    #     for i in layers_to_plot:
    #         if 1 <= i < self.num_layers:
    #             grads = self.gradients[f"W{i}"].flatten()
    #             plt.figure()
    #             plt.hist(grads, bins=30, edgecolor='black')
    #             plt.title(f"Distribusi Gradien Bobot untuk Layer {i}")
    #             plt.xlabel("Nilai Gradien")
    #             plt.ylabel("Frekuensi")
    #             plt.show()
    #         else:
    #             print(f"Layer {i} tidak valid untuk distribusi gradien.")

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Model berhasil disimpan di {filename}")

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.__dict__ = pickle.load(f)
        print(f"Model berhasil dimuat dari {filename}")