# 📈📰 Sentiment Analysis of Financial News

## 🚀 Project Overview
This project classifies sentiment in financial news using NLP techniques. It compares traditional models like **Naive Bayes** with neural networks (**Feed-Forward & GRU**) and uses **BERT embeddings** for improved accuracy. The project includes in-depth data analysis and various text preprocessing strategies and was completed for the "Natural Language Processing" course at TU Berlin.

---

## 🧰 Technologies Used
- **Language**: 🐍 Python 3
- **Environment**: 📓 Jupyter Notebook
- **Key Libraries**:
  - `pandas` for data manipulation
  - `nltk` for natural language processing
  - `scikit-learn` for machine learning models
  - `tensorflow` (Keras) for neural networks
  - `matplotlib` & `seaborn` for data visualization

---

## ✨ Key Features
- 🔎 **In-depth Data Analysis**: Explore data distribution and key vocabulary patterns for each sentiment class.
- 🧹 **Advanced Text Preprocessing**: Tokenization, stemming, and stop-word removal pipelines.
- ⚖️ **Model Comparison**: Naive Bayes vs. Feed-Forward Neural Network.
- 🧠 **Bonus**: GRU-based RNN with pre-trained BERT embeddings for enhanced classification.
- 🤝 **Word Similarity**: Word–word similarity via a Pointwise Mutual Information (PMI) matrix.

---

## 📊 Results
Models were evaluated on accuracy and F1-score, with neural networks generally outperforming the Naive Bayes classifier. For detailed results, hyperparameter tuning, and error analysis, see the full project report at `report/report.pdf`.

---

## 🧪 How to Run This Project
Follow these steps to set up and run the project locally.

1️⃣ Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/nlp-sentiment-analysis.git
cd nlp-sentiment-analysis
```

2️⃣ Create and activate a virtual environment

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

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Launch the Jupyter Notebook
```bash
jupyter notebook Project1_Gilbert_Sahitaj.ipynb
```

---

## 📎 Notes
- Replace `YOUR_USERNAME` in the clone URL with your GitHub username if you forked this repo.
- If you face issues with `nltk` resources, you may need to download them in a Python shell:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```
- If you see a tqdm/ipywidgets warning in notebooks, install widgets support:
```bash
pip install ipywidgets
```

---

## 📄 License
This project is licensed under the terms of the license found in the `LICENSE` file.
