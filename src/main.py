import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ffnn import FFNN
from activation import relu, softmax
from loss import categorical_cross_entropy_loss, d_categorical_cross_entropy_loss
from initialization import init_weights_normal

def load_and_prepare_data():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data
    y = np.array(mnist.target).astype(int).reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def run_experiment(X_train, y_train, X_test, y_test):
    layers = [128, 64, 10]  
    activations = [relu, softmax, relu]  
    init_params = {'mean': 0, 'variance': 0.1, 'seed': 42} 
    batch_size = 32
    epochs = 50
    learning_rate = 0.01

    reg_type = None  
    lambda_val = 0  

    models = []
    histories = []

    print(f"\nTraining model with {reg_type if reg_type else 'No'} regularization (λ={lambda_val})")
    model = FFNN(
        layers=[X_train.shape[1]] + layers,
        activations=activations,
        loss_func=categorical_cross_entropy_loss,
        loss_grad=d_categorical_cross_entropy_loss,
        init_method=init_weights_normal,
        init_params=init_params,
        reg_type=reg_type,
        lambda_reg=lambda_val
    )

    history = model.train(
        X_train, y_train,
        X_test, y_test,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        verbose=1
    )

    models.append(model)
    histories.append(history)

    return models, histories

def analyze_and_plot(models, histories, layers):
    print("\nComparing prediction results:")
    for i, reg_type in enumerate(['No regularization']):
        model = models[i]
        y_pred = model.forward(X_test)
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == true_labels)
        print(f"{reg_type}: Accuracy = {accuracy:.4f}")

    plt.figure(figsize=(10, 5))
    for i, reg_type in enumerate(['No reg']):
        plt.plot(histories[i]["train_loss"], label=f"{reg_type} - Train")
        plt.plot(histories[i]["val_loss"], label=f"{reg_type} - Val", linestyle='--')
    plt.title('Training and Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(15, 10))
    for l in range(1, len(layers)+1): 
        for i, reg_type in enumerate(['No reg']):
            plt.subplot(len(layers), len(['No reg']), (l-1)*len(['No reg']) + i + 1)
            weights = models[i].weights[l-1].flatten()
            plt.hist(weights, bins=30, alpha=0.7)
            plt.title(f"Layer {l} - {reg_type}")
            plt.xlabel("Weight Value")
    plt.tight_layout()
    plt.show()

    for i, reg_type in enumerate(['No reg']):
        print(f"\nDisplaying model graph for {reg_type}:")
        models[i].display_model_graph()

    for i, reg_type in enumerate(['No reg']):
        print(f"\nPlotting weight and gradient distributions for {reg_type}:")
        models[i].plot_weight_and_gradient_distribution([1, 2])  

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    models, histories = run_experiment(X_train, y_train, X_test, y_test)

    layers = [128, 64, 10]
    analyze_and_plot(models, histories, layers)
