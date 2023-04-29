# File Upload Interface

The File Upload Interface is a user-friendly web application that preprocesses, extracts, and augments text. It uses Flask and Flask-SocketIO for real-time communication, and HTML, CSS, and JavaScript for the interface. The application can convert files to text, extract keywords and entities, and apply text augmentation techniques. It exports the processed data in JSONL.

## Getting Started

### Prerequisites

Before running the application, make sure to have the following dependencies installed:

- Python 3.8
- Flask 2.1.1
- Werkzeug 2.1.2
- summa 1.2.0
- python-docx 0.8.11
- spacy 3.1.3
- nltk 3.6.6
- jsonlines 2.0.0
- pdfminer-six 20221105
- python-magic 0.4.27
- gevent 22.10.2
- pdfplumber 0.9.0
- scikit-learn 1.2.2
- transformers 4.28.1
- sentencepiece 0.1.98
- sacremoses 0.0.53

### Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/file-upload-interface.git
cd file-upload-interface
```

2. Install the dependencies:

```
pip install -r requirements.txt
```

### Usage

To start the application, run:

```
python app.py
```

This will start the Flask server, which can be accessed by navigating to http://localhost:5000 in a web browser.

#### Uploading a file

To upload a file, select a file and choose the appropriate use case from the dropdown menu. The file should be a PDF, DOCX, TXT, or DOC file and not exceed 10 MB in size. Once the file is uploaded, the application will convert it to plain text and perform the selected use case. The processed data will be exported as a JSONL file, which can be downloaded by clicking on the "Download" button.

#### Text Classification

The text classification use case classifies the text into predefined categories. The application uses the BERT model to classify the text.

#### Sentiment Analysis

The sentiment analysis use case determines the sentiment of the text. The application uses the BERT model to analyze the sentiment.

#### Entity Extraction

The entity extraction use case identifies the named entities in the text. The application uses the spaCy model to extract entities.

#### Chatbot

The chatbot use case generates responses to user input. The application uses the GPT-2 model to generate responses.

#### Product Description

The product description use case generates product descriptions based on input attributes. The application uses a template-based approach to generate descriptions.

## Contributing

Contributions are welcome! If you find any issues or would like to contribute new features or enhancements, please submit a pull request.

## License

This project is licensed under the GNU license. See the [LICENSE](LICENSE) file for details.