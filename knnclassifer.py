import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def extract_features(image_path):
    # Read image in grayscale format
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize image to 64x64 pixels
    image_resized = cv2.resize(image, (64, 64))

    # Flatten image into a 1D array
    features = image_resized.flatten()

    return features

def read_dataset(dataset_path):
    images = []
    labels = []

    # Define the list of disease categories
    disease_categories = ['V_Calcified Roots','V_Long Roots','V_Multi-curvature','V_Narrow Canals','V_Abnormal anatomy','V_Curve canals','V_Bifurcated canals']

    # Loop over each disease category
    for category in disease_categories:
        # Get the path to the directory containing images for this category
        category_path = os.path.join(dataset_path, category)

        # Loop over each image file in this category directory
        for image_file in os.listdir(category_path):
            if image_file.endswith('.bmp') :  # Read only bmp files
                # Get the path to the image file
                image_path = os.path.join(category_path, image_file)

                # Extract features from the image
                image_features = extract_features(image_path)

                # Add the image features and label to the dataset
                images.append(image_features)
                labels.append(category)

    return images, labels

# Load the dataset
dataset_path = r'G:/Vspace/dental data/Sorted images(Combined)'
images, labels = read_dataset(dataset_path)

# Convert the dataset to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Convert category names to integer labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.1,random_state=75)

# Create the KNN classifier and train it on the training set
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Test the classifier on the testing set
y_pred = knn.predict(X_test)

# Convert integer labels back to category names
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)


# Load the new image
new_image_path = r'G:\Vspace\dental data\New Sorting\Calcified Roots\s20150213_185330_0000.bmp'
new_image_features = extract_features(new_image_path)

# Predict the category of the new image
new_image_features_reshaped = new_image_features.reshape(1, -1)
predicted_label_encoded = knn.predict(new_image_features_reshaped)

# Convert the predicted integer label back to the category name
predicted_label = label_encoder.inverse_transform(predicted_label_encoded)

# Print the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test_labels, y_pred_labels))

print('The predicted disease category of the new image is:', predicted_label)

from sklearn.metrics import accuracy_score, classification_report
category_names = label_encoder.classes_
accuracies = []
for category in category_names:
    y_test_category = (y_test_labels == category)
    y_pred_category = (y_pred_labels == category)
    accuracy = accuracy_score(y_test_category, y_pred_category)
    accuracies.append(accuracy)
    print(f"Accuracy for {category}: {accuracy:.2f}")

# Plot the accuracies for each category
import matplotlib.pyplot as plt


plt.bar(category_names, accuracies)
plt.title("Accuracy for each category")
plt.xlabel("Category")
plt.ylabel("Accuracy")
plt.show()