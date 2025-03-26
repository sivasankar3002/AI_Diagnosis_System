# AI-Powered Disease Diagnosis System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)


An AI-powered web application built with **Streamlit** to predict various diseases such as **Diabetes**, **Heart Disease**, **Lung Cancer**, and **Parkinson’s Disease** using machine learning models.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset & Models](#dataset--models)
6. [Contributing](#contributing)
7. [License](#license)

---

## Overview
This project implements an AI-based diagnosis system that allows users to predict the likelihood of specific diseases based on input features. The application uses pre-trained machine learning models and provides an intuitive user interface powered by **Streamlit**.

---

## Features
- **Disease Prediction Options**:
  - Diabetes Prediction
  - Heart Disease Prediction
  - Lung Cancer Prediction
  - Parkinson’s Disease Prediction
- **User-Friendly Interface**:
  - Dropdown menu to select disease prediction type.
  - Input fields tailored to each disease's dataset features.
- **Probabilistic Output**:
  - Displays both binary predictions (Yes/No) and probability scores.
- **Scalable Design**:
  - Easily extendable to include additional diseases or models.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Streamlit (`pip install streamlit`)
- Scikit-learn, XGBoost, Joblib, and other dependencies

### Steps to Set Up
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/AI-Diagnosis-System.git
   cd AI-Diagnosis-System
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-Trained Models**:
   - Place the following pre-trained models in the root directory:
     - `diabetes_model.pkl`
     - `heart_model_xgb.joblib`
     - `lung_cancer_model.pkl`
     - `parkinsons_model.pkl`

4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

---

## Usage

### Selecting a Prediction Option
1. Open the app in your browser after running `streamlit run app.py`.
2. Use the sidebar dropdown menu to select a disease prediction option:
   - Diabetes Prediction
   - Heart Disease Prediction
   - Lung Cancer Prediction
   - Parkinson’s Disease Prediction

### Inputting Data
- Fill out the required input fields based on the selected disease.
- For example:
  - **Diabetes**: Age, BMI, Blood Pressure, etc.
  - **Heart Disease**: Age, Chest Pain Type, Cholesterol, etc.
  - **Parkinson’s**: Vocal measurement features like MDVP:Fo(Hz), Jitter(%), etc.

### Viewing Results
- Click the "Predict" button to get the prediction and probability score.
- Example Output:
  ```
  **Prediction:** Diabetes
  **Probability:** 85.42%
  ```

---

## Dataset & Models

### Datasets Used
- **Diabetes**: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Heart Disease**: [Heart Disease UCI Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Lung Cancer**: [Custom Lung Cancer Dataset](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)
- **Parkinson’s**: [UCI Parkinson’s Dataset](https://archive.ics.uci.edu/ml/datasets/Parkinsons)

### Model Details
| Disease          | Model Used           | Accuracy |
|-------------------|----------------------|----------|
| Diabetes         | Random Forest       | ~90%     |
| Heart Disease    | XGBoost             | ~85%     |
| Lung Cancer      | Random Forest       | ~89%     |
| Parkinson’s      | Random Forest       | ~90%     |

---

## Contributing
We welcome contributions to improve this project! Here’s how you can contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with detailed changes.

### Reporting Issues
If you encounter any bugs or have suggestions, please open an issue in the repository.

---

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgments
- **Streamlit**: For providing an easy-to-use framework for building interactive apps.
- **Scikit-learn & XGBoost**: For robust machine learning libraries.
- **Datasets**: Thanks to UCI Machine Learning Repository and Kaggle for open-source datasets.

---

## Contact
For questions or feedback, feel free to reach out:
- Email: sivasankar3002@gmail.com
- GitHub: [Your GitHub Profile](https://github.com/sivasankar3002)

---
