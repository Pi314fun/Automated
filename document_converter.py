import os
import pdfplumber
from docx import Document
import sys
import magic


def get_file_extension(file_path):
    magic_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "magic", "libmagicwin64-master", "magic.mgc")
    os.environ['MAGIC_FILE'] = magic_path
    file_mime = magic.Magic(mime=True).from_file(file_path)

    if file_mime == 'application/pdf':
        return 'pdf'
    elif file_mime == 'application/msword':
        return 'doc'
    elif file_mime == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return 'docx'
    elif file_mime == 'text/plain':
        return 'txt'
    else:
        return None


def convert_pdf_to_text(file_path: str) -> str:
    with pdfplumber.open(file_path) as pdf:
        text = '\n'.join(page.extract_text() for page in pdf.pages)
    return text


def convert_docx_to_text(file_path):
    document = Document(file_path)
    text = ''
    for paragraph in document.paragraphs:
        text += paragraph.text + '\n'
    return text


def convert_file_to_text(file_path):
    file_extension = get_file_extension(file_path)
    text = ""

    def read_text_file(file_path, encoding):
        with open(file_path, "r", encoding=encoding) as file:
            return file.read()

    if file_extension == "pdf":
        text = convert_pdf_to_text(file_path)

    elif file_extension == "docx":
        text = convert_docx_to_text(file_path)

    elif file_extension == "txt":
        encodings = ['utf-8', 'ISO-8859-1', 'cp1252']
        success = False
        for encoding in encodings:
            try:
                text = read_text_file(file_path, encoding)
                success = True
                break
            except UnicodeDecodeError:
                pass
        if not success:
            raise ValueError("Unable to decode file with supported encodings")

    else:
        raise ValueError("Unsupported file format")

    return text


def remove_null_characters(text):
    return text.replace('\x00', '')


def save_converted_text_to_file(text, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Convert the input file to text
    converted_text = convert_file_to_text(input_file)

    # Remove null characters from the text
    converted_text = remove_null_characters(converted_text)

    # Save the converted text to the output file
    save_converted_text_to_file(converted_text, output_file)
