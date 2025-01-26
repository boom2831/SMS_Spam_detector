# SMS_Spam_Detector

This is a SMS Spam Detector built using **Natural Language Processing (NLP)** techniques and **Streamlit** for the web interface. The model identifies whether an SMS message is spam or not.

## Features

- Detects spam messages using a pre-trained machine learning model.
- Text preprocessing with tokenization, stopword removal, and lemmatization.
- Interactive web interface for easy SMS input and spam detection.

## Tech Stack

- **Python**
- **Streamlit**: For the web app interface.
- **scikit-learn**: For the machine learning model.
- **nltk**: For text preprocessing.
- **pickle**: For saving and loading the model and vectorizer.

## Setup Instructions

1. Clone this repository:
   git clone https://github.com/boom2831/SMS_Spam_detector.git
   cd sms-spam-detector

2. Install the required dependencies:
    pip install -r requirements.txt

3. Download the necessary NLTK datasets:
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

4. Ensure the following files are in the project directory:
    vectorizer.pkl: The saved CountVectorizer object.
    model.pkl: The saved machine learning model.

5. Run the app:
    streamlit run app.py

6. Open your browser and go to http://localhost:8501 to access the app.

## How It Works

Input: Enter your SMS message in the input box.
Processing: The SMS is preprocessed (lowercased, tokenized, stopwords removed, and lemmatized).
Prediction: The pre-trained machine learning model predicts whether the message is spam or not.
Output: The result is displayed on the screen.

## Example
Input: Congratulations! You have won a $1000 gift card. Reply YES to claim.

Output: This SMS is SPAM

Input: Hi John, can we reschedule the meeting to 3 PM?

Output: This SMS is not SPAM

## Project Structure

sms-spam-detector/
│
├── app.py              # Streamlit app code
├── vectorizer.pkl      # Saved CountVectorizer object
├── model.pkl           # Trained machine learning model
├── requirements.txt    # List of dependencies
├── README.md           # Project documentation


## Model Details
The machine learning model was trained using:

RandomForestClassifier: Achieving an accuracy of 98% on the test dataset.
Feature Extraction: Character-level n-grams (3 to 4 characters).
Preprocessing: Lowercasing, tokenization, stopword removal, and lemmatization.

## Acknowledgments
NLTK for text preprocessing tools.
scikit-learn for machine learning.
Streamlit for the interactive web application framework.
