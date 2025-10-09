# land-classification-using-deep-learning
Deep learning project for land classification using satellite imagery with CNN and Vision Transformer models in Keras and PyTorch.


# 🌍 Land Classification Using Deep Learning

This project applies Convolutional Neural Networks (CNNs) and Vision Transformers (ViT) to classify geospatial satellite images into different land types such as agricultural and non agricultural.

---

## 🧠 Project Overview

The goal of this project is to develop a deep learning pipeline that can:
- Load and preprocess satellite image datasets
- Train and evaluate CNN and Vision Transformer models
- Compare performance using accuracy, F1-score, and AU-ROC metrics

---

## ⚙️ Tech Stack

- **Python**
- **Keras / TensorFlow**
- **PyTorch**
- **NumPy, Pandas, Matplotlib**
- **Scikit-learn**

---

## 📂 Folder Structure

land-classification-using-deep-learning/  
│  
├── data/ # Image datasets or download links    
├── notebooks/ # Jupyter notebooks for model development    
├── models/ # Saved model files (.keras, .pth)    
├── reports/ # Evaluation results and summaries  
├── scripts/ #Python scripts 
├── requirements.txt # Dependencies    
└── README.md # Project overview    

## 🚀 How to Run

bash     
git clone https://github.com/<your-username>/land-classification-using-deep-learning.git   
cd land-classification-using-deep-learning   
pip install -r requirements.txt   
jupyter notebook    

## Run all Scripts

To run all scripts all at once, which will take some time.  
Run: ./scripts/run_all.py  

Note: To generate a better model change the value of     
n_epochs = 3 to (30 or more)    
