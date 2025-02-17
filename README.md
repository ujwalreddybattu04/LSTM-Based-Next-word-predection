# LSTM-Based-Next-word-predection
Tis project aims to develop a deep-learning model for predecting the next word in a given sequences of words.The model is built using (LSTM)networks,which are well suited for sequence predection tasks
1.Data collection:we use the text of shakespeares Hamlet asour dataset 
2.Data preprocessing: The text data is tokenized converted into sequences and padded to ensure uniform input lenghts.The Sequences are then split into training and testing sets
3.model building:An LSTM model is constructed with an emddbedding layer,two lstm layers and a dense output with a softmax activation function to predeict the word probability.
===================================================================================================================================================================
To install dependencies:
cd venv
pip install -r requirements.txt
To run this project:
cd LSTM-RNN
streamlit run app.py
===================================================================================================================================================================
Project structure
|-- next_word_lstm.h5        # Trained LSTM model
|-- tokenizer.pickle         # Tokenizer for text processing
|-- app.py                   # Streamlit application
|-- requirements.txt         # Required dependencies
|-- README.md                # Project documentation
=====================================================================================================================================================================
This project is a Next Word Prediction application built using an LSTM (Long Short-Term Memory) model. The application provides multiple features, including:

✅ Multi-word prediction (predicts the next 3 words)
✅ Speech-to-Text input using a microphone
✅ Text-to-Speech to read out the predicted words
✅ Word Definition Lookup using a dictionary API
✅ Multi-Language Translation using Google Translate API
✅ Dark Mode UI for a modern and clean look
======================================================================================================================================================================
Features:
🎤 Features

📝 Next Word Prediction
Uses a pre-trained LSTM model to predict the next 3 words.

🎙️ Speech-to-Text
Allows users to input text using voice commands.
Uses the SpeechRecognition library.

🔊 Text-to-Speech
Converts predicted text into speech output.
Uses pyttsx3 for offline TTS.

📚 Word Definition Lookup
Fetches word meanings using Dictionary API.

🌍 Multi-Language Translation
Translates input text into multiple languages (French, Spanish, German, Hindi, Chinese).

🌙 Dark Mode UI
Enhanced UI for a sleek, modern look using Streamlit's styling
====================================================================================================================================================================
To clone the repo
git clone https://github.com/ujwalreddybattu04/LSTM-Based-Next-word-predection.git
====================================================================================================================================================================
word predector interface
![image](https://github.com/user-attachments/assets/8b1bcf8a-0894-4737-9f7f-978367f15f4f)

📦 Dependencies

Python 3.8+
TensorFlow
Streamlit
SpeechRecognition
Pyttsx3
Googletrans
Requests
Numpy
Pickle
I got an accuracy of 83.5

