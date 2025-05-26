# 🦷 Teeth Diseases Classification

## 📌 Objective

This project aims to develop a powerful and efficient **computer vision model for classifying dental diseases** from clinical images. Accurate classification of dental diseases such as **CaS, CoS, Gum, MC, OC, OLP, and OT** is vital for supporting AI-driven healthcare systems that assist in early diagnosis and treatment planning, ultimately **enhancing diagnostic accuracy and improving patient outcomes**.

The pipeline includes:

* Image preprocessing and augmentation
* Dataset visualization
* Baseline and advanced model development
* Model Optimization and Transfer Learning
* Deployment using Streamlit

---

## 🗂️ Dataset

The dataset used is a subset of the **MOD (Medical Oral Dataset)**, which contains images labeled across 7 classes:

* **CaS** – Canker Sores
* **CoS** – Cold Sores 
* **Gum** – Gingivostomatitis
* **MC** – Mouth Cancer
* **OC** – Oral Cancer 
* **OLP** – Oral Lichen Planus
* **OT** – Oral Thrush 

### Dataset Splitting

* **60%** for training: 3,087 images
* **20%** for validation: 1,028 images
* **20%** for testing: 1,028 images
* **Total before augmentation**: 5,143 images

---

## 🧼 1. Preprocessing

Preprocessing steps ensure the dataset is clean and ready for modeling:

* **Normalization**: Scales pixel values to \[0, 1]
* **Resizing**: All images are resized uniformly
* **Augmentation**: Includes rotation, zoom, brightness adjustments, and flipping to improve robustness and reduce overfitting

---

## 📊 2. Visualization

To better understand the data:

* Distribution of each disease category is visualized
* Pre- and post-augmentation image samples are displayed
* These visualizations confirm data balance and preprocessing effectiveness

---

## 🧠 3. Model Development & Training

### 🏗️ Baseline CNN Model

* Built using TensorFlow/Keras from scratch
* Consists of multiple Conv2D, MaxPooling, Dropout, and Dense layers
* Trained for multi-class classification with categorical cross-entropy loss

✅ **Achieved 97% accuracy on the test set**, establishing a strong baseline.

---

## 🔁 4. Fine-tuning with MobileNetV2

To improve accuracy and efficiency, **MobileNetV2** was fine-tuned:

* Pretrained weights used with custom classification head
* Retrained on our dataset with fine-tuning enabled

📈 **Results**:

* **Accuracy:** 99.7%
* **Model complexity reduced by 81.4%**, making it significantly more lightweight and faster for deployment

---

## 🚀 5. Deployment

The final model was deployed using **Streamlit** for easy and interactive use.

### 🔬 Deployment Testing:

* Evaluated on the 1,028-image test set
* **Correctly diagnosed 1,026 out of 1,028 images**

The Streamlit app enables users to upload dental images and get **instant disease predictions**, making it suitable for both clinical and educational settings.


