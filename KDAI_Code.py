import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
DATASET_PATH = "/kaggle/input/eye-balnc/eye balnc data"



dataset = tf.keras.preprocessing.image_dataset_from_directory("/kaggle/input/eye-balnc/eye balnc data")
print("Classes:", dataset.class_names)

image_count = sum(1 for _ in dataset.unbatch())
print("Total images:", image_count)


# 1. Load Entire Dataset
full_dataset = image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=None  # Load all images into a single dataset
)

# 2. Convert dataset to NumPy arrays
images = []
labels = []

for img, label in full_dataset:
    images.append(img.numpy())
    labels.append(label.numpy())

images = np.stack(images)
labels = np.array(labels)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

# 4. Convert back to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# 5. Add Augmentation to training data
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# 6. Shuffle, batch, prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)


# 7. Build Fine-Tuned ResNet50 Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes
])


# 8. Compile and Train
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)


# 9. Plot Training History
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Test Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training & Test Accuracy')
plt.grid(True)
plt.show()


# Plot Loss Curves
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Test Loss')
plt.grid(True)
plt.show()




**TESTING**

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

# Set the path to your test image
test_image_path = '/kaggle/input/eye-balnc/eye balnc data/Normal/image_1.jpg'  # Replace with actual path

# Load and preprocess the image
img = image.load_img(test_image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = preprocess_input(img_array)        # Same preprocessing as ResNet50

# Predict class
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions, axis=1)[0]

# Map index to class name
class_names = dataset.class_names
predicted_class_name = class_names[predicted_class_index]

# Show result
plt.imshow(img)
plt.title(f"Predicted: {predicted_class_name}")
plt.axis('off')
plt.show()

print(f"Predicted class: {predicted_class_name} (Index: {predicted_class_index})")
# 3. Show result
print(f"Predicted class: {class_names[predicted_class]}")
print(f"Confidence: {confidence:.2f}")


# Plot Loss Curves
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Test Loss')
plt.grid(True)
plt.show()


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Get true and predicted labels
y_true = []
y_pred = []

for images, labels in test_dataset:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt

# Generate classification report
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

# Convert to DataFrame for plotting
df = pd.DataFrame(report).transpose()
df = df.loc[class_names, ['precision', 'recall', 'f1-score']]

# Plot
df.plot(kind='bar', figsize=(10, 6))
plt.title('Classification Report Metrics by Class')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import numpy as np

# Binarize the labels for multi-class ROC
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_binarized.shape[1]

# Get prediction probabilities from model
y_score = model.predict(X_test)

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(6, 4))
colors = ['blue', 'green', 'red']
class_labels = class_names  # e.g., ['Keratoconus', 'Normal', 'Suspect']

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f"{class_labels[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Multi-Class')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Flatten images from (height, width, channels) to (features,)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Create and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_flat, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test_flat)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Flatten images from (height, width, channels) to (features,)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Create and train the model
svm_model = SVC(kernel='linear', random_state=42)  # You can change the kernel to 'rbf' or 'poly' for different SVM types
svm_model.fit(X_train_flat, y_train)

# Predict and evaluate
y_pred = svm_model.predict(X_test_flat)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy * 100:.2f}%")


# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train Logistic Regression Model
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train_flat, y_train)

# Predict on the test set
y_pred = log_reg_model.predict(X_test_flat)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")


import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Flatten the image data (if working with images)
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten training images
X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Flatten test images

# Optionally, scale the features for better performance
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)

# Create and train the XGBoost classifier
xg_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xg_model.fit(X_train_flat, y_train)

# Predict on the test set
y_pred = xg_model.predict(X_test_flat)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy * 100:.2f}%")
