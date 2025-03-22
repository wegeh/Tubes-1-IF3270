import numpy as np
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from activation import linear, d_linear, relu, d_relu, sigmoid, d_sigmoid, tanh, d_tanh, softmax, d_softmax, leaky_relu, d_leaky_relu, elu, d_elu
from regularization import l1_regularization, l2_regularization
from tqdm import tqdm  

class FFNN:
    def __init__(self, layers, activations, loss_func, loss_grad, init_method, init_params, reg_type=None, lambda_reg=0.0):
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
        self.reg_type = reg_type
        self.lambda_reg = lambda_reg

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
            elif act_func == softmax:
                delta *= d_softmax(z)
            elif act_func == leaky_relu:
                delta *= d_leaky_relu(z)
            elif act_func == elu:
                delta *= d_elu(z)
            
            # buat softmax belom tau mo gimana ges

            a_prev = self.a_values[i]
            self.gradients[f"W{i+1}"] = np.dot(a_prev.T, delta) / m
            self.gradients[f"b{i+1}"] = np.sum(delta, axis=0, keepdims=True) / m

            if self.reg_type == 'L1':
                self.gradients[f"W{i+1}"] += self.lambda_reg * np.sign(self.weights[i]) / m
            elif self.reg_type == 'L2':
                self.gradients[f"W{i+1}"] += self.lambda_reg * self.weights[i] / m

            if i != 0:
                delta = np.dot(delta, self.weights[i].T)

    def update_weights(self, learning_rate):
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * self.gradients[f"W{i+1}"]
            self.biases[i] -= learning_rate * self.gradients[f"b{i+1}"]

    def calculate_loss(self, X, y):
        """Calculate loss with optional regularization term
        
        Parameters:
            X: Input features
            y: Target values
        """
        y_pred = self.forward(X) 
        loss = self.loss_func(y, y_pred)
        
        if self.reg_type == 'L1':
            l1_cost = 0
            for w in self.weights:
                l1_cost += self.lambda_reg * np.sum(np.abs(w)) / X.shape[0]
            loss += l1_cost
        elif self.reg_type == 'L2':
            l2_cost = 0
            for w in self.weights:
                l2_cost += 0.5 * self.lambda_reg * np.sum(np.square(w)) / X.shape[0]
            loss += l2_cost
                
        return loss
    
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
                loss = self.calculate_loss(X_batch, y_pred)
                epoch_loss += loss.item() if hasattr(loss, 'item') else loss

                self.backward(X_batch, y_batch)
                self.update_weights(learning_rate)
            
            avg_train_loss = epoch_loss / num_batches
            history["train_loss"].append(avg_train_loss)
            
            val_loss = self.calculate_loss(X_val, y_val)
            history["val_loss"].append(val_loss)
            
            if verbose:
                tqdm.write(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        return history

    def display_model_graph(self, max_neurons_per_layer=10, node_size=300, 
                      weight_precision=2, min_display_value=1e-4):
        """
        Display a visualization of the neural network with weights, gradients and bias.
        """

        G = nx.DiGraph()

        selected_neurons = {}
        for layer in range(self.num_layers):
            layer_size = self.layers[layer]
            if layer_size <= max_neurons_per_layer:
                selected_neurons[layer] = list(range(layer_size))
            else:
                indices = np.linspace(0, layer_size-1, max_neurons_per_layer).astype(int)
                selected_neurons[layer] = list(indices)
        
        layer_colors = ['#8A2BE2', '#4682B4', '#3CB371', '#FFD700']  
        
        for layer in range(self.num_layers):
            color = layer_colors[min(layer, len(layer_colors)-1)]
            layer_name = "Input" if layer == 0 else "Output" if layer == self.num_layers-1 else f"Hidden {layer}"
            
            for neuron in selected_neurons[layer]:
                G.add_node(f"L{layer}_N{neuron}", 
                        layer=layer, 
                        color=color,
                        layer_name=layer_name)
                
            # Add bias node
            if layer < self.num_layers - 1:  
                G.add_node(f"L{layer}_B", 
                        layer=layer, 
                        color="#D3D3D3",  
                        layer_name=layer_name,
                        is_bias=True)
        
        max_abs_weight = 0
        for i in range(1, self.num_layers):
            for j in selected_neurons[i-1]:
                for k in selected_neurons[i]:
                    max_abs_weight = max(max_abs_weight, abs(self.weights[i-1][j, k]))
        
        for i in range(1, self.num_layers):
            for j in selected_neurons[i-1]:
                for k in selected_neurons[i]:
                    weight = self.weights[i-1][j, k]
                    grad = self.gradients[f"W{i}"][j, k]

                    if abs(weight) < min_display_value and abs(grad) < min_display_value:
                        continue

                    weight_str = f"{weight:.{weight_precision}f}" if abs(weight) >= min_display_value else "0"
                    grad_str = f"{grad:.{weight_precision}f}" if abs(grad) >= min_display_value else "0"
                    
                    if weight < 0:
                        color = 'red'
                    else:
                        color = 'green'
                    
                    width = 0.5 + 3.0 * (abs(weight) / max_abs_weight) if max_abs_weight > 0 else 0.5
                    
                    G.add_edge(f"L{i-1}_N{j}", f"L{i}_N{k}",
                            weight=weight,
                            gradient=grad,
                            width=width,
                            color=color,
                            weight_str=weight_str,
                            grad_str=grad_str,
                            fontsize=8,
                            edge_type="weight")
            
            for k in selected_neurons[i]:
                bias = self.biases[i-1][0, k]
                bias_grad = self.gradients[f"b{i}"][0, k]
                
                if abs(bias) < min_display_value and abs(bias_grad) < min_display_value:
                    continue
                    
                bias_str = f"{bias:.{weight_precision}f}" if abs(bias) >= min_display_value else "0"
                bias_grad_str = f"{bias_grad:.{weight_precision}f}" if abs(bias_grad) >= min_display_value else "0"
                
                color = 'orange' if bias >= 0 else 'purple'
                
                width = 0.5 + 2.0 * (abs(bias) / max_abs_weight) if max_abs_weight > 0 else 0.5
                
                G.add_edge(f"L{i-1}_B", f"L{i}_N{k}",
                        weight=bias,
                        gradient=bias_grad,
                        width=width,
                        color=color,
                        weight_str=bias_str,
                        grad_str=bias_grad_str,
                        fontsize=8,
                        edge_type="bias")

        pos = {}
        for layer in range(self.num_layers):
            x = layer * 5  
            neurons = selected_neurons[layer]
            n_neurons = len(neurons)
            
            y_spacing = 1.5
            total_height = (n_neurons - 1) * y_spacing
            
            for idx, neuron in enumerate(neurons):
                y_pos = -total_height/2 + idx * y_spacing
                pos[f"L{layer}_N{neuron}"] = (x, y_pos)
            
            if layer < self.num_layers - 1:
                bias_y = -total_height/2 - 2
                pos[f"L{layer}_B"] = (x, bias_y)
        
        fig, ax = plt.subplots(figsize=(15, 10), facecolor='white')
        
        regular_nodes = [n for n in G.nodes() if not G.nodes[n].get('is_bias', False)]
        bias_nodes = [n for n in G.nodes() if G.nodes[n].get('is_bias', False)]
        
        nx.draw_networkx_nodes(G, pos,
                            nodelist=regular_nodes,
                            node_color=[G.nodes[n]['color'] for n in regular_nodes],
                            node_size=node_size,
                            ax=ax)
        
        if bias_nodes:
            nx.draw_networkx_nodes(G, pos,
                                nodelist=bias_nodes,
                                node_color='lightgray',
                                node_size=node_size * 0.8,
                                node_shape='s',
                                ax=ax)
        
        for u, v, data in G.edges(data=True):
            edge_color = data['color']
            width = data['width']
            nx.draw_networkx_edges(G, pos,
                                edgelist=[(u, v)],
                                width=width,
                                edge_color=edge_color,
                                arrowsize=10,
                                arrowstyle='->' if width > 1 else '-',
                                connectionstyle="arc3,rad=0.1",
                                alpha=0.7,
                                ax=ax)
        
        node_labels = {n: n.split('_')[1] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, 
                            labels=node_labels,
                            font_size=10,
                            font_weight='bold')

        edge_labels = {}
        for u, v, data in G.edges(data=True):
            edge_labels[(u, v)] = f"w: {data['weight_str']}\ng: {data['grad_str']}"
        
        for edge, label in edge_labels.items():
            edge_data = G.get_edge_data(*edge)
            x1, y1 = pos[edge[0]]
            x2, y2 = pos[edge[1]]
            
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2 + 0.1 
            
            if edge_data['edge_type'] == 'weight':
                bg_color = 'lightgreen' if edge_data['weight'] >= 0 else 'lightcoral'
            else:  
                bg_color = 'lightyellow' if edge_data['weight'] >= 0 else 'lavender'
                
            ax.text(mid_x, mid_y, label,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8,
                bbox=dict(
                    facecolor=bg_color,
                    edgecolor='black',
                    boxstyle='round,pad=0.3',
                    alpha=0.8
                ))
        
        for layer in range(self.num_layers):
            layer_name = "Input Layer" if layer == 0 else "Output Layer" if layer == self.num_layers - 1 else f"Hidden {layer} Layer"
            neuron_count = self.layers[layer]
            title = f"{layer_name}\n({neuron_count} neurons)"
            
            x = layer * 5
            if selected_neurons[layer]:
                y_top = max([pos[f"L{layer}_N{n}"][1] for n in selected_neurons[layer]]) + 2
            else:
                y_top = 2
                
            color = layer_colors[min(layer, len(layer_colors)-1)]
            ax.text(x, y_top, title,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=12,
                fontweight='bold',
                bbox=dict(
                    facecolor=color,
                    edgecolor='black',
                    boxstyle='round,pad=0.5',
                    alpha=0.7
                ))
        
        import matplotlib.patches as mpatches
        legend_elements = [
            mpatches.Patch(facecolor='green', alpha=0.7, label='Positive Weight'),
            mpatches.Patch(facecolor='red', alpha=0.7, label='Negative Weight'),
            mpatches.Patch(facecolor='orange', alpha=0.7, label='Positive Bias'),
            mpatches.Patch(facecolor='purple', alpha=0.7, label='Negative Bias'),
            mpatches.Patch(facecolor='lightgray', label='Bias Node')
        ]
        
        for i, color in enumerate(layer_colors[:self.num_layers]):
            layer_name = "Input" if i == 0 else "Output" if i == self.num_layers - 1 else f"Hidden {i}"
            legend_elements.append(mpatches.Patch(facecolor=color, alpha=0.7, label=f"{layer_name} Layer"))
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.figtext(0.01, 0.01, 
                "Notation on connections:\nw: weight value\ng: gradient value", 
                fontsize=10,
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))
        
        ax.axis('off')
        plt.tight_layout()
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