# Importing necessary libraries for anomaly detection with autoencoders
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Setting the seaborn style and a seed for reproducibility
sns.set(style="whitegrid")
np.random.seed(203)

# Loading the Credit Card Fraud dataset from the specified path
# Note: Adjust the path if necessary
data = pd.read_csv(r"C:\Users\Patron\Downloads\archive\creditcard.csv")

# Preprocessing the 'Time' column: converting time to hours
data["Time"] = data["Time"].apply(lambda x: x / 3600 % 24)

# Basic statistics to understand the dataset's structure and summary
print(data.describe())

# Visualizing the distribution of the 'Class' column to see how imbalanced it is
vc = data['Class'].value_counts().to_frame().reset_index()
vc['percent'] = vc["Class"].apply(lambda x: round(100*float(x) / len(data), 2))
vc = vc.rename(columns={"index": "Target", "Class": "Count"})
print(vc)

# Sampling the data to create a balanced dataset for training
# Using a small subset of non-fraud cases to prevent class imbalance issues during training
non_fraud = data[data['Class'] == 0].sample(1000)
fraud = data[data['Class'] == 1]
df = pd.concat([non_fraud, fraud]).sample(frac=1).reset_index(drop=True)

# Splitting features (X) and labels (Y) from the sampled data
X = df.drop(['Class'], axis=1).values
Y = df["Class"].values

# Feature scaling to normalize the data, which is crucial for training the autoencoder
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to visualize the data in 2D space using t-SNE with summaries
def tsne_plot(x1, y1, name="tsne_graph.png", title="", subtitle=""):
    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(x1)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth=1, alpha=0.8, label='Non Fraud')
    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth=1, alpha=0.8, label='Fraud')
    
    plt.title(title)
    plt.text(0.95, 0.01, subtitle,
             verticalalignment='bottom', horizontalalignment='right', 
             transform=plt.gca().transAxes, 
             color='black', fontsize=12)
    plt.legend(loc='best')
    plt.savefig(name)
    plt.close()  # Close the figure to free up memory

# Visualizing the original data distribution before training
tsne_plot(X_scaled, Y, "original.png", 
          title="Original Data Distribution",
          subtitle="Green = Non-Fraud, Red = Fraud\nFraudulent transactions are mixed with non-fraudulent ones,\nmaking detection difficult.")

# Building the Autoencoder model for anomaly detection
input_layer = Input(shape=(X_scaled.shape[1],))

# Encoding part: compressing the input data into a lower-dimensional space
encoded = Dense(100, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoded = Dense(50, activation='relu')(encoded)

# Decoding part: reconstructing the compressed data back to the original dimensions
decoded = Dense(50, activation='tanh')(encoded)
decoded = Dense(100, activation='tanh')(decoded)

# Output layer: final reconstruction of the input
output_layer = Dense(X_scaled.shape[1], activation='relu')(decoded)

# Defining the model and compiling it with the Adam optimizer and MSE loss function
autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Splitting the dataset into training and testing sets
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Training the autoencoder model on the training data
# The autoencoder is trained to minimize the reconstruction error
autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, shuffle=True, validation_data=(X_test, X_test), verbose=1)

# Using the trained autoencoder to reconstruct the entire dataset
X_pred = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)  # Calculating the reconstruction error

# Setting a threshold for anomaly detection based on the 95th percentile of the reconstruction error
threshold = np.percentile(mse, 95)

# Classifying transactions as anomalies if the reconstruction error exceeds the threshold
y_pred = np.where(mse > threshold, 1, 0)

# Evaluating the model's performance
print("Confusion Matrix:")
print(confusion_matrix(Y, y_pred))

print("\nClassification Report:")
print(classification_report(Y, y_pred))

print("\nAccuracy Score:", accuracy_score(Y, y_pred))

# Visualizing the data distribution after anomaly detection
tsne_plot(X_scaled, y_pred, "anomalies.png", 
          title="Detected Anomalies after Autoencoder",
          subtitle="Green = Predicted Non-Fraud, Red = Predicted Fraud\nAutoencoder has separated some transactions as anomalies.")

# Function to visualize the data in 2D space using PCA with summaries
def pca_plot(x1, y1, name="pca_graph.png", title="", subtitle=""):
    pca = PCA(n_components=2)
    X_t = pca.fit_transform(x1)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth=1, alpha=0.8, label='Non Fraud')
    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth=1, alpha=0.8, label='Fraud')
    
    plt.title(title)
    plt.text(0.95, 0.01, subtitle,
             verticalalignment='bottom', horizontalalignment='right', 
             transform=plt.gca().transAxes, 
             color='black', fontsize=12)
    plt.legend(loc='best')
    plt.savefig(name)
    plt.close()  # Close the figure to free up memory

# Alternative Visualization: PCA plot after autoencoder
pca_plot(X_scaled, y_pred, "pca_anomalies.png", 
          title="PCA: Detected Anomalies after Autoencoder",
          subtitle="Green = Predicted Non-Fraud, Red = Predicted Fraud\nPCA applied after autoencoder to visualize detected anomalies.")

# Fourth Visualization: Reconstruction Error Distribution
plt.figure(figsize=(12, 8))
plt.hist(mse, bins=50, color='blue', alpha=0.7)
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend(loc='upper right')
plt.savefig("reconstruction_error.png")
plt.close()  # Close the figure to free up memory

# Final Summary: Observations, Results, and Conclusions
plt.figure(figsize=(12, 8))
plt.text(0.5, 0.9, "Anomaly Detection Summary", horizontalalignment='center', fontsize=20, fontweight='bold')
plt.text(0.5, 0.7, "Method: Autoencoder trained on normal transactions to detect anomalies.", horizontalalignment='center', fontsize=14)
plt.text(0.5, 0.6, "Challenges: High class imbalance and overlapping features.", horizontalalignment='center', fontsize=14)
plt.text(0.5, 0.5, f"Results: Accuracy = {accuracy_score(Y, y_pred):.2f}, See classification report for details.", horizontalalignment='center', fontsize=14)
plt.text(0.5, 0.4, "Observations: See visualizations for detected anomalies and error distribution.", horizontalalignment='center', fontsize=14)
plt.text(0.5, 0.3, "Conclusion: Autoencoder shows promise, but may need further optimization.", horizontalalignment='center', fontsize=14)
plt.axis('off')
plt.savefig("summary.png")
plt.close()  # Close the figure to free up memory

print("All visualizations have been saved as PNG files in the current directory.")

# This code is adapted from the work of Harsh Mehta, with modifications for TensorFlow 2.x and improved visualizations.