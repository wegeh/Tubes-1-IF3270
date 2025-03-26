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
    X = mnist.data
    y = np.array(mnist.target).astype(int).reshape(-1, 1)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def run_experiment(X_train, y_train, X_test, y_test):
    layers = [784, 128, 64, 10]  
    activations = [relu,  relu, softmax]  
    init_params = {'lower_bound': 0, 'upper_bound': 0.5, 'seed': 42}
    batch_size = 32
    epochs = 20
    learning_rate = 0.01

    models = []
    histories = []

    # Model TANPA RMSNorm
    print("\nTraining model TANPA RMSNorm")
    model_no_norm = FFNN(
        layers=layers,
        activations=activations,
        loss_func=categorical_cross_entropy_loss,
        loss_grad=d_categorical_cross_entropy_loss,
        init_method=init_weights_he,
        init_params=init_params,
        reg_type=None,
        lambda_reg=0.0,
        use_rmsnorm=False
    )
    history_no_norm = model_no_norm.train(
        X_train, y_train, X_test, y_test,
        batch_size, epochs, learning_rate, verbose=1
    )
    models.append(model_no_norm)
    histories.append(history_no_norm)

    # Model DENGAN RMSNorm
    print("\nTraining model DENGAN RMSNorm")
    model_rmsnorm = FFNN(
        layers=layers,
        activations=activations,
        loss_func=categorical_cross_entropy_loss,
        loss_grad=d_categorical_cross_entropy_loss,
        init_method=init_weights_he,
        init_params=init_params,
        reg_type=None,
        lambda_reg=0.0,
        use_rmsnorm=True
    )
    history_rmsnorm = model_rmsnorm.train(
        X_train, y_train, X_test, y_test,
        batch_size, epochs, learning_rate, verbose=1
    )
    models.append(model_rmsnorm)
    histories.append(history_rmsnorm)

    return models, histories

def analyze_and_plot(models, histories, X_test, y_test):
    labels = ['Tanpa RMSNorm', 'Dengan RMSNorm']
    print("\nPerbandingan hasil prediksi:")
    for i, label in enumerate(labels):
        model = models[i]
        y_pred = model.forward(X_test)
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == true_labels)
        print(f"{label}: Accuracy = {accuracy:.4f}")

    # Plot grafik loss pelatihan dan validasi
    plt.figure(figsize=(10, 5))
    for i, label in enumerate(labels):
        plt.plot(histories[i]["train_loss"], label=f"{label} - Train")
        plt.plot(histories[i]["val_loss"], label=f"{label} - Val", linestyle='--')
    plt.title('Perbandingan Loss Pelatihan dan Validasi')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot distribusi bobot (misal untuk layer hidden pertama)
    plt.figure(figsize=(15, 5))
    for i, label in enumerate(labels):
        plt.subplot(1, len(labels), i+1)
        weights = models[i].weights[0].flatten()  # layer 1 (setelah input)
        plt.hist(weights, bins=30, alpha=0.7)
        plt.title(f"Distribusi Bobot - {label} - Layer 1")
        plt.xlabel("Nilai Bobot")
    plt.tight_layout()
    plt.show()

    # Plot distribusi gradien (misal untuk layer hidden pertama)
    plt.figure(figsize=(15, 5))
    for i, label in enumerate(labels):
        plt.subplot(1, len(labels), i+1)
        grads = models[i].gradients.get("W1", np.array([])).flatten()
        plt.hist(grads, bins=30, alpha=0.7)
        plt.title(f"Distribusi Gradien - {label} - Layer 1")
        plt.xlabel("Nilai Gradien")
    plt.tight_layout()
    plt.show()

    # Tampilkan graf struktur model (jika diinginkan)
    for i, label in enumerate(labels):
        print(f"\nMenampilkan graf struktur model untuk {label}:")
        models[i].display_model_graph()

    # Plot distribusi bobot dan gradien menggunakan metode yang disediakan
    for i, label in enumerate(labels):
        print(f"\nPlot distribusi bobot dan gradien untuk {label}:")
        models[i].plot_weight_and_gradient_distribution([1])

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    models, histories = run_experiment(X_train, y_train, X_test, y_test)
    analyze_and_plot(models, histories, X_test, y_test)
