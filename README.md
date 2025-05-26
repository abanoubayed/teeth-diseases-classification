# 🦷 Teeth Classification Project

## 📌 Objective

This project aims to develop a robust computer vision solution for **classifying dental images into 7 distinct categories**. This classification is part of our AI-driven healthcare initiative, designed to **enhance diagnostic accuracy** and **improve patient outcomes** through advanced machine learning techniques.

The full pipeline includes:

* Image preprocessing
* Dataset visualization and augmentation
* Model development and training
* Performance evaluation

---

## 🧼 1. Preprocessing

The preprocessing step ensures the dataset is in optimal condition for training:

* **Normalization**: Pixel values are scaled to a consistent range to stabilize training.
* **Resizing**: All images are resized to a uniform size for model compatibility.
* **Augmentation**: Includes horizontal flips, rotations, zoom, and brightness adjustments to artificially expand the dataset and improve generalization.

Before and after augmentation samples are displayed to verify transformation quality.

---

## 📊 2. Visualization

Understanding the dataset distribution is key to robust training. This includes:

* Visual inspection of a sample of raw and augmented images.
* **Class distribution plots** to detect any imbalance across the 7 categories.

This step helps guide preprocessing and training decisions.

---

## 🧠 3. Model Architecture & Training

* The model is built from scratch using **TensorFlow (Keras)**.
* Architecture includes:

  * Multiple **Conv2D** layers with ReLU activations
  * **MaxPooling** and **Dropout** layers to reduce overfitting
  * A final **Dense softmax layer** for multiclass classification
* The model is compiled with:

  * `categorical_crossentropy` loss
  * `Adam` optimizer
  * Accuracy as the main metric

Training is performed using:

* 70% of the data for training
* 15% for validation
* 15% for testing

**Callbacks** like EarlyStopping and ModelCheckpoint are used to prevent overfitting and retain the best model.

---

## 📈 Results & Evaluation

* Training and validation accuracy and loss curves are plotted to monitor learning.
* The test set is used to evaluate the final model performance.
* A classification report and confusion matrix help assess per-class performance.

---

