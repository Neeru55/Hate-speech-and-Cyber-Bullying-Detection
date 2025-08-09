# Hate Speech and Cyberbullying Detection using BERT

A state-of-the-art Natural Language Processing (NLP) system for detecting cyberbullying and hate speech in social media text using BERT.

## 🎯 Project Overview

This project implements a deep learning-based solution for detecting various types of cyberbullying and hate speech in social media content. Using the BERT transformer model, it can classify text into multiple categories of harmful content with high accuracy.

## 📚 Dataset

The project uses the [Cyberbullying Dataset by Saurabh Shahane](https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset) from Kaggle, containing over 45,000 labeled tweets across different categories of cyberbullying.

### Categories:
- Age-based cyberbullying
- Ethnicity-based cyberbullying
- Gender-based cyberbullying
- Religion-based cyberbullying
- Other types of cyberbullying
- Not cyberbullying

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/hate-speech-detection.git
cd hate-speech-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 🚀 Usage

### Training the Model
```bash
python train.py
```

### Running the Web App
```bash
streamlit run app.py
```

## 📁 Project Structure

```
hate-speech-detection/
├── data/                   # Dataset directory
├── models/                 # Saved models
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── app.py                 # Streamlit web application
├── requirements.txt
└── README.md
```

## 🧪 Model Architecture

- Base Model: BERT (bert-base-uncased)
- Classification Head: Dense layer with softmax activation
- Output: 6 classes (different types of cyberbullying)

## 📊 Performance Metrics

The model achieves:
- Accuracy: ~90%
- F1-Score: ~0.89
- Precision: ~0.88
- Recall: ~0.90

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Kaggle](https://www.kaggle.com/) for hosting the dataset
- [Streamlit](https://streamlit.io/) for the web framework 