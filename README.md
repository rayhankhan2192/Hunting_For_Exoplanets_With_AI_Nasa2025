## 🚀 Try SpectraAI  

[http://104.154.252.161:5173/](http://104.154.252.161:5173/)

# Frontend Code
[https://github.com/rayhankhan2192/SpectraAI-WebApp](https://github.com/rayhankhan2192/SpectraAI-WebApp)

## 🔗 Live Demo  
[![Live Demo](https://img.shields.io/badge/Try-SpectraAI-blue?style=for-the-badge&logo=vercel)](http://203.190.12.138:5174/)

# How to use the AI
### 1. Hunt The Exoplanate

1. Download the KOI, K2 Test data from GitHub
2. Go to SpectraAI [http://104.154.252.161:5173/](http://104.154.252.161:5173/)
3. Click HUNT EXOPLANATE or Start Hunt
4. Select the mision KOI, K2
5. Select Model for predicting, use XGB Boost for better accuracy
6. Scroll down see the perforances and Click export for download the predicted CSV

### 2. Update the model with Train + Predicted data

1. Download the KOI Train CSV
1. Goto DataFusion [http://104.154.252.161:5173/merge](http://104.154.252.161:5173/merge)
3. Select Train and your predicted file
4. Click merge then use
5. Train your new powerful model

## Features updated
```bash
1. Launch new featrues for K2 satellite pipeline, robust Train, prediction with dynamic setup

2. Updated KOI pipeline to load all artifacts dynaically

3. Add new ML model, SVM, LogisticRegression, GradientBoostingClassifier
```  


## Install Requirements

### 1. Clone the Repository:
```bash
git clone git@github.com:rayhankhan2192/Hunting_For_Exoplanets_With_AI_Nasa2025.git
cd Hunting_For_Exoplanets_With_AI_Nasa2025
```

### 2. Create a Virtual Environment:

```bash
python -m venv venv
```

### 3.Activate the Virtual Environment:
```bash
.\venv\Scripts\activate
```

### 4.Install Dependencies:
```bash
pip install -r requirements.txt
```

### 5. Run the Training Script:
```bash
python train.py --data-path "path/to/k2_data.csv" --satellite "K2/TOI/TESS" --model "xgb/rf/decisiontree"
```