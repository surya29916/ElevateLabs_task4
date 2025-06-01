# ElevateLabs_Task4

## 📊 Dataset Overview

The dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses. Each row represents a tumor sample with 30 numerical features, and the target variable indicates whether the tumor is malignant or benign.

---

## Steps Performed

### 1. Import and Preprocess the Dataset
- Loaded the dataset using `pandas`.
- Dropped irrelevant columns like `id` and `Unnamed: 32`.
- Converted the `diagnosis` column:
  - `M` (Malignant) → `1`
  - `B` (Benign) → `0`
- Split the data into features (`X`) and target (`y`).

### 2. Split Data and Standardize Features
- Used `train_test_split()` to divide the data into:
  - 80% training data
  - 20% test data
- Applied `StandardScaler` to normalize the feature values to have mean 0 and standard deviation 1.

### 3. Fit a Logistic Regression Model
- Used `LogisticRegression` from `sklearn.linear_model` to train the model.
- Trained the model using scaled training data.

### 4. Evaluate the Model
- Made predictions on the test data.
- Evaluated the model using:
  - **Confusion Matrix**
  - **Classification Report** (Precision, Recall, F1-score)
  - **ROC-AUC Score** (Area under the ROC curve)
  - **ROC Curve** plot to visualize performance across thresholds

#### 🔄 Sigmoid Function

The **sigmoid function** is used in Logistic Regression to map predicted values to probabilities between 0 and 1. It transforms the output of the linear model into a probability score.

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

- \( z \) is the linear combination of features: \( z = w_1x_1 + w_2x_2 + ... + w_nx_n + b \)
- The output \( \sigma(z) \) represents the probability that a given input belongs to the **positive class** (e.g., malignant tumor)

#### ⚖️ Threshold Tuning

- By default, Logistic Regression uses a threshold of **0.5**:
  - If predicted probability ≥ 0.5 → class 1 (Malignant)
  - If predicted probability < 0.5 → class 0 (Benign)
- The threshold can be **adjusted** to:
  - Increase **sensitivity/recall** (detect more positives)
  - Reduce **false positives** or **false negatives**
---

## Interpretation

- A **high ROC-AUC** indicates strong model performance.
- The **confusion matrix** helps identify false positives and false negatives.
- The **sigmoid curve** helps understand the decision boundary and how predictions are made from probabilities.
- Tuning the **threshold** can help balance between sensitivity (recall) and specificity.

---

## 🛠️ Tools & Libraries Used

- Python
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn

---
