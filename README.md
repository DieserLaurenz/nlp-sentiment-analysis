# Sentiment Analysis of Financial News

## Project Overview
This project classifies sentiment in financial news using NLP techniques. It compares traditional models like **Naive Bayes** with neural networks (**Feed-Forward & GRU**) and uses **BERT embeddings** for improved accuracy. The project includes in-depth data analysis and various text preprocessing strategies and was completed for the "Natural Language Processing" course at TU Berlin.

---

## Technologies Used
- **Python 3**
- **Jupyter Notebook**
- **Key Libraries:**
    - `pandas` for data manipulation
    - `nltk` for natural language processing
    - `scikit-learn` for machine learning models
    - `tensorflow` (Keras) for neural networks
    - `matplotlib` & `seaborn` for data visualization

---

## Directory Structure
nlp-sentiment-analysis/
├── nlp_sentiment_analysis.ipynb
├── report/
│   └── report.pdf
├── .gitignore
├── Sentences_50Agree.txt
├── LICENSE
├── README.md
└── requirements.txt

---

## Key Features
- **In-depth Data Analysis:** Explored data distribution and key vocabulary patterns for each sentiment class.
- **Advanced Text Preprocessing:** Implemented various pipelines including tokenization, stemming, and stop-word removal.
- **Model Comparison:** Evaluated and compared the performance of a Naive Bayes classifier against a Feed-Forward Neural Network.
- **Bonus Task:** Developed a Recurrent Neural Network (GRU) with pre-trained BERT embeddings for enhanced sentiment classification.
- **Word Similarity:** Calculated word-word similarity using a Pointwise Mutual Information (PMI) matrix.

---

## Results
The models were evaluated on accuracy and F1-score, with the neural network models generally outperforming the Naive Bayes classifier. A detailed breakdown of the results, hyperparameter tuning, and error analysis can be found in the full project report located at `report/report.pdf`.

---

## How to Run This Project

Follow these steps to set up and run the project locally.

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/nlp-sentiment-analysis.git
cd nlp-sentiment-analysis
```

2. Create and activate a virtual environment

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the Jupyter Notebook
```bash
jupyter notebook notebooks/nlp_sentiment_analysis.ipynb
```