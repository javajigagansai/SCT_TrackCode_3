import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

# Constants
IMAGE_SIZE = (64, 64)
CAT_DIR = "./PetImages/Cat"
DOG_DIR = "./PetImages/Dog"

dataset = []
labels = []

# Load Cat Images
print("📥 Loading dataset...")
print("Loading Cat images:", end=" ")
cat_images = os.listdir(CAT_DIR)[:41]  # Limit to 41
for img_name in tqdm(cat_images):
    img_path = os.path.join(CAT_DIR, img_name)
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMAGE_SIZE)
        dataset.append(img.flatten())
        labels.append(0)  # Cat = 0
    except:
        continue

# Load Dog Images
print("Loading Dog images:", end=" ")
dog_images = os.listdir(DOG_DIR)[:41]  # Limit to 41
for img_name in tqdm(dog_images):
    img_path = os.path.join(DOG_DIR, img_name)
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMAGE_SIZE)
        dataset.append(img.flatten())
        labels.append(1)  # Dog = 1
    except:
        continue

print(f"✅ Loaded {len(dataset)} images.")

# Check if dataset is usable
if len(dataset) < 10:
    print("⚠️ Warning: Very few images loaded. Accuracy may be unreliable.")

# Convert to NumPy arrays
X = np.array(dataset) / 255.0  # Normalize
y = np.array(labels)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predictions & Evaluation
y_pred = clf.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

print("📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
