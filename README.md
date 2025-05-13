
# Gender Classification using Conventional Machine Learning

This project demonstrates a simple machine learning pipeline to classify gender based on facial features using **conventional ML algorithms** such as:

- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors (KNN)

The classification task uses a structured dataset and applies preprocessing, training, evaluation (including confusion matrix), and manual input testing to simulate a real-world use case.

---

## 📂 Dataset

The dataset contains the following features:
- `long_hair`
- `forehead_width_cm`
- `forehead_height_cm`
- `nose_wide`
- `nose_long`
- `lips_thin`
- `distance_nose_to_lip_long`
- **Target:** `gender` (Male / Female)

Data source: `gender.csv`

---

## 🧠 Models Used

- **SVM (Support Vector Machine)**
- **Decision Tree Classifier**
- **K-Nearest Neighbor Classifier (KNN)**

Each model is trained using `train_test_split`, and performance is evaluated with:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

---

## 🛠️ Workflow

1. Data Preprocessing
   - Handling duplicates
   - Label encoding
   - Feature scaling
2. Model Training (SVM, DT, KNN)
3. Evaluation & Comparison
   - Classification reports
   - Confusion matrix visualization
4. Manual Input Testing (real-case scenario)

---

## 📊 Confusion Matrix

Visual representation of true vs predicted labels to understand classification performance.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_conf_matrix(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
```

---

## 🧪 Manual Testing Example

Testing the model with a single, unseen input:

```python
sample = [[1, 13.0, 5.8, 0, 0, 1, 0]]
```

Predicted consistently as **Male** by all models.

> 🟡 Note: You might get a warning like `UserWarning: X does not have valid feature names...`. To fix this:

```python
import pandas as pd
sample = pd.DataFrame([[1, 13.0, 5.8, 0, 0, 1, 0]], columns=X.columns)
sample_scaled = scaler.transform(sample)
model.predict(sample_scaled)
```

---

## 📎 Dependencies

```bash
scikit-learn
numpy
pandas
matplotlib
seaborn
```

> You can run this notebook using Jupyter or Google Colab.

---

## 📌 Author

**Sulthon Kaf**  
_Computer Science Student | Machine Learning Enthusiast_

---

## 🌟 Acknowledgement

This project was developed as part of a university assignment to apply binary/multiclass classification using conventional ML techniques.
