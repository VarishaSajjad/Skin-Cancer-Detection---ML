# Skin Cancer Detection using Deep Learning

This project focuses on the automated detection of skin cancer using dermoscopic images and deep learning techniques. The goal is to classify skin lesions as benign or malignant to support early diagnosis and decision-making in healthcare applications.

## Dataset
The model was trained and evaluated using a publicly available dermoscopic image dataset.

- Dataset: HAM10000 / ISIC Skin Cancer Dataset  
- Classes: Benign, Malignant  
- Data Type: Dermoscopic skin lesion images  

Dataset link:  
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

## Approach
A Convolutional Neural Network (CNN) was used for binary classification of skin lesions. The model learns visual patterns from medical images to distinguish between benign and malignant cases.

Key steps:
- Image preprocessing and normalization  
- CNN-based feature extraction  
- Binary classification using deep learning  
- Model evaluation using accuracy and confusion matrix  

## Results
The model achieved strong performance on the validation dataset.

- Validation Accuracy: ~87%  
- The confusion matrix shows effective classification of both benign and malignant cases with balanced precision and recall.
- Training and validation loss curves indicate stable learning with no severe overfitting.

Confusion matrix summary:
- True Benign: 328  
- True Malignant: 246  
- False Positives: 32  
- False Negatives: 54  

These results demonstrate the effectiveness of deep learning techniques for medical image classification.

## Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- OpenCV  

## How to Run
1. Clone the repository:
2. Install dependencies:
3. Open the notebook:
4. Run all cells sequentially.

## Future Improvements
- Apply transfer learning (ResNet, VGG, EfficientNet)
- Improve recall for malignant cases
- Extend to multi-class skin lesion classification
- Deploy the model as a web or mobile application

## Disclaimer
This project is for educational and research purposes only and is not intended for clinical use.

