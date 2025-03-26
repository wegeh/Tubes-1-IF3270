# main.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ffnn import FFNN
from activation import relu, softmax
from loss import categorical_cross_entropy_loss, d_categorical_cross_entropy_loss
from initialization import init_weights_normal, init_weights_he
from tqdm import tqdm

def load_and_prepare_data():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(int)
    # print(mnist)
    # print("x: ",X)
    # print("y: ", y)

    # df = pd.DataFrame(X)
    # df['label'] = y
    # print(df.head())
    # Standarisasi fitur
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # One-hot encode label
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_onehot = encoder.fit_transform(y.to_numpy().reshape(-1, 1))


    # Membagi data menjadi training dan validation
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def run_experiment(X_train, y_train, X_test, y_test):
    layers = [64, 64, 10]  
    activations = [relu, relu, relu]  
    init_params = {'lower_bound': 0, 'upper_bound': 0.1, 'seed': 42} 
    batch_size = 32
    epochs = 20
    learning_rate = 0.01

    reg_types = [None, 'L1', 'L2']
    lambda_values = [0, 0.01, 0.01]

    models = []
    histories = []

    for reg_type, lambda_val in zip(reg_types, lambda_values):
        print(f"\nTraining model with {reg_type if reg_type else 'No'} regularization (λ={lambda_val})")
        model = FFNN(
            layers=[X_train.shape[1]] + layers,
            activations=activations,
            loss_func=categorical_cross_entropy_loss,
            loss_grad=d_categorical_cross_entropy_loss,
            init_method=init_weights_he,
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

def analyze_and_plot(models, histories):
    layers = [64, 64, 10]
    print("\nComparing prediction results:")
    for i, reg_type in enumerate(['No regularization', 'L1 regularization', 'L2 regularization']):
        model = models[i]
        y_pred = model.forward(X_test)
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == true_labels)
        print(f"{reg_type}: Accuracy = {accuracy:.4f}")

    # Plot loss comparison
    plt.figure(figsize=(10, 5))
    for i, reg_type in enumerate(['No reg', 'L1 reg', 'L2 reg']):
        plt.plot(histories[i]["train_loss"], label=f"{reg_type} - Train")
        plt.plot(histories[i]["val_loss"], label=f"{reg_type} - Val", linestyle='--')
    plt.title('Training and Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot weight distributions for each regularization type
    plt.figure(figsize=(15, 10))
    for l in range(1, len(layers)+1):  # For each layer
        for i, reg_type in enumerate(['No reg', 'L1 reg', 'L2 reg']):
            plt.subplot(len(layers), len(['No reg', 'L1 reg', 'L2 reg']), (l-1)*len(['No reg', 'L1 reg', 'L2 reg']) + i + 1)
            weights = models[i].weights[l-1].flatten()
            plt.hist(weights, bins=30, alpha=0.7)
            plt.title(f"Layer {l} - {reg_type}")
            plt.xlabel("Weight Value")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 10))
    for l in range(1, len(layers)+1):  # For each layer
        for i, reg_type in enumerate(['No reg', 'L1 reg', 'L2 reg']):
            plt.subplot(len(layers), len(['No reg', 'L1 reg', 'L2 reg']), (l-1)*len(['No reg', 'L1 reg', 'L2 reg']) + i + 1)
            gradients = models[i].gradients[f"W{l}"].flatten()
            plt.hist(gradients, bins=30, alpha=0.7)
            plt.title(f"Layer {l} - {reg_type if reg_type else 'No'} reg")
            plt.xlabel("Gradient Value")
            
    plt.tight_layout()
    plt.savefig('gradient_distribution.png')

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    models, histories = run_experiment(X_train, y_train, X_test, y_test)

    analyze_and_plot(models, histories)