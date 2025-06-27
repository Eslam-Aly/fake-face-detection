# 🧠 Fake Face Detection using Xception and Transfer Learning

This project uses deep learning and transfer learning with the **Xception** model to detect whether a face image is **real or fake (AI-generated)**. It was developed as part of a Machine Learning course assignment.

<code>
  fake_face_detection/
│
├── notebooks/
│   └── fake_face_detector.ipynb       # Main Jupyter Notebook
│
├── requirements.txt                   # Python dependencies
├── .gitignore
└── README.md                          # This file
</code>

---

## 🧬 Model Details

- **Base model**: Xception from Keras applications  
- **Strategy**: Transfer learning with full fine-tuning  
- **Optimizer**: Adam with low learning rate `1e-5`  
- **Architecture**:  
  - Rescaling layer  
  - Xception base (unfrozen)  
  - GlobalAveragePooling2D  
  - Dropout (0.5)  
  - Dense (128) + ReLU  
  - Dropout (0.3)  
  - Dense (1) + Sigmoid  

---

## 📈 Performance

- **Training Accuracy**: ~98%  
- **Validation Accuracy**: ~80%  
- **Test Metrics**:  
  - Accuracy, F1-Score, Confusion Matrix  
  - Plots for 3 correct and 3 incorrect predictions  

---

## 📊 Evaluation

- Metrics are visualized in the notebook  
- Includes:  
  - Accuracy and Loss curves  
  - Classification report  
  - 3 correct / 3 incorrect predictions from unseen test data  
<code>
 pip install -r requirements.txt
jupyter notebook notebooks/fake_face_detector.ipynb
</code>

