# COS30049-Group2-AI_Governance
Group project for COS30049 Computing Technology Innovation Project

# Environment Setup (Requires Miniconda or Anaconda)

# Create the environment with the exact versions used in the report
conda create -n ai-detect python=3.13 -y
conda activate ai-detect

# Core libraries
conda install -c conda-forge pandas=3.0.6 numpy=2.5.3 scikit-learn=1.9.1 lightgbm=4.7.0 joblib=1.6.0 -y

#  plotting + notebook (EDA only)
conda install -c conda-forge matplotlib seaborn jupyter -y

# Data
The data folder is gitignored because the raw CSVs are too large for GitHub (one exceeds the 100MB hard limit). Download the datasets yourself and place them in the data folder before running the file:

1. DAIGT V2 Train Dataset: https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset
2. AI vs Human Text Classification Dataset 2026: https://www.kaggle.com/datasets/alitaqishah/ai-vs-human-text-classification-dataset-2026
3. AI Vs Human Text: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text


# Data processing
Run the notebook top to botton:

jupyter notebook eda.ipynb

# Train the models
Run in this order – each step uses the outputs of the previous one. All scripts use fixed random seeds.

# Baseline sentence classifiers (optional – used for the comparison in Table 3)
python classifier.py --model logreg     # -> models/clf_logreg.joblib
python classifier.py --model svc        # -> models/clf_svc.joblib

# Final sentence classifier: LightGBM
#     early stopping uses 5,000 documents held back from the training split
python lgbm_classifier.py               # -> models/classifier.joblib

# GMM clustering on AI training sentences only (k = 6, full covariance)
#     also saves the BIC curve and cluster profile tables
python gmm.py --k 6                     # -> models/gmm.joblib            

# Document-level regressor, fitted on the VALIDATION split
#     (documents the classifier and GMM never saw), then evaluated on test + holdout_2026
python regressor.py                     # -> models/regressor.joblib

# Make Predictions
Predict a single text:
python pipeline.py --text "Paste the text you want to check here."          

# Reference
Used for feature engineering

1. https://academic.oup.com/dsh/advance-article/doi/10.1093/llc/fqag064/8714041
2. https://arxiv.org/abs/2303.03697
3. https://www.researchgate.net/publication/398588043_Feature-Based_Detection_of_AI-Generated_Text_An_Analysis_of_Stylometric_and_Perplexity_Markers_in_Contemporary_Large_Language_Models

