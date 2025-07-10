import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === Configuration === #
DATA_DIR = './PetImages'
CATEGORIES = ['Cat', 'Dog']
IMG_SIZE = 32  # Reduced size for efficiency

# Check if dataset path exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"❌ Dataset folder not found at: {DATA_DIR}")
else:
    print(f"📁 Found dataset at: {DATA_DIR}")

# === Load Images === #
def load_data():
    data = []
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        label = CATEGORIES.index(category)
        for img_name in tqdm(os.listdir(path), desc=f"Loading {category} images"):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"⚠️ Failed to load {img_path}")
                    continue
                resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                flattened = resized.flatten()
                data.append([flattened, label])
            except Exception as e:
                print(f"⚠️ Failed to process {img_path}: {str(e)}")
                continue
    return data

print("📥 Loading dataset...")
dataset = load_data()
print(f"✅ Loaded {len(dataset)} images.")

# Validate dataset size
if len(dataset) < 100:
    raise ValueError(f"❌ Insufficient data: Only {len(dataset)} images loaded.")

# Prepare data
X = np.array([features for features, _ in dataset]) / 255.0  # Normalize
y = np.array([label for _, label in dataset])

# === Train/Test Split === #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train SVM === #
print("🧠 Training SVM model...")
model = LinearSVC(max_iter=10000)
model.fit(X_train, y_train)

# === Evaluate Model === #
print("📊 Evaluating model...")
y_pred = model.predict(X_test)
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))
print("🎯 Accuracy Score:", accuracy_score(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()