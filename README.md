# Logistic-Regression-SVM-Naive-Bayes-Coursework
Academic Machine Learning coursework focusing on Logistic Regression, Support Vector Machine (SVM), Naive Bayes. Includes model evaluation on [Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).

## 📚 Project Overview
- **Dataset Preparation**: Splitting data into training and testing sets.
- **Model Training**: Implementing Logistic Regression, SVM with different kernels (Linear and RBF), and Naive Bayes classifiers.
- **Evaluation**: Assessing models based on accuracy and confusion matrices.

## 📊 Dataset
The dataset used in this project is `heart.csv`. It contains clinical data related to heart disease, with columns representing features such as age, sex, cholesterol levels, and more. The target variable, target, indicates the presence (1) or absence (0) of heart disease.

## ✨ Features
- **Preprocessing**: Splitting the dataset into training (90%) and testing (10%) sets.
- **Model Training**:
  - Logistic Regression with 1000 iterations.
  - Support Vector Machines with Linear and RBF kernels, using different regularization parameter values (C=1, C=10, C=20).
  - Naive Bayes classifier.
- **Model Evaluation**: Accuracy scores and confusion matrices are used to assess performance.

## 📚 Results
The results of the models are evaluated using:
- **Accuracy**: Overall predictive performance of the model.
- **Confusion Matrix**: Details of true positives, true negatives, false positives, and false negatives for each model.

**Summary**
- **Logistic Regression**: Achieved an accuracy of XX% (example).
- **SVM (Linear, RBF)**: Achieved varying accuracies depending on the kernel and C values.
- **Naive Bayes**: Achieved an accuracy of YY% (example).

## 🚀 How to Use
1. **Clone the repository**:
  ```bash
  git clone https://github.com/ioannantreki/Logistic-Regression-SVM-Naive-Bayes-Coursework.git
  ```

2. **Upload the `heart.csv` dataset to the specified directory**.

3. **Run the Code**:
- Execute the script in Google Colab or a similar environment.
- Ensure dependencies such as `pandas`, `sklearn`, and `numpy` are installed.

4. **Evaluate Results**:
- Review the printed accuracy scores and confusion matrices in the console.

## ℹ️ Contact
For any inquiries or collaboration requests, please reach out via GitHub or email at ioannadreki31@gmail.com.
