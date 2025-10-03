## Try Frontend

```bash
http://203.190.12.138:5174/

past in broswer
````


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