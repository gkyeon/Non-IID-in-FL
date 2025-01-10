import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Define CNN model structure
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Load data
def load_data(partition):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() #fashion_mnist data load
    x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    # Split data for federated learning
    x_train_partition = x_train[partition * 30000:(partition + 1) * 30000]
    y_train_partition = y_train[partition * 30000:(partition + 1) * 30000]
    x_test_partition = x_test[partition * 5000:(partition + 1) * 5000]
    y_test_partition = y_test[partition * 5000:(partition + 1) * 5000]
    return x_train_partition, y_train_partition, x_test_partition, y_test_partition

# Function to load model weights and evaluate performance
def evaluate_performance(round_number):
    # Load parameters and metrics
    param_file = f"round-{round_number}-parameters.npz"
    metric_file = f"round-{round_number}-metrics.npz"

    params = np.load(param_file)
    metrics = np.load(metric_file)

    # Load model structure and set weights
    model = create_cnn_model()
    weights = [params[f"arr_{i}"] for i in range(len(params.files))]
    model.set_weights(weights)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load test data
    partition = 0  # Assuming the partition you used in training
    x_train, y_train, x_test, y_test = load_data(partition)

    # Evaluate performance
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Round {round_number} - Loss: {loss}, Accuracy: {accuracy}")

    # Predict on test data
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Compute Confusion Matrix and Classification Report
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    cr = classification_report(y_true_classes, y_pred_classes, target_names=[str(i) for i in range(10)])

    print(f"Round {round_number} - Classification Report:\n{cr}")

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Round {round_number}')
    plt.show()

    # Display sample images
    num_samples = 10 
    random_indices = np.random.choice(x_test.shape[0], num_samples, replace=False)
    sample_images = x_test[random_indices]
    sample_labels = y_test[random_indices]

    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
        plt.title(np.argmax(sample_labels[i]))
        plt.axis('off')
    plt.show()

# Evaluate for rounds 1 to 20
for round_number in range(0,100): #config
    try:
        evaluate_performance(round_number)
    except Exception as e:
        print(f"Error evaluating round {round_number}: {e}")