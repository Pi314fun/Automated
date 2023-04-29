# app.py
import traceback
from document_converter import convert_file_to_text, save_converted_text_to_file
from use_cases import text_classification_pipeline, sentiment_analysis_pipeline, entity_extraction_pipeline, chatbot_pipeline, product_description_pipeline
from datetime import datetime
from werkzeug.utils import secure_filename
import os
from flask_socketio import SocketIO, emit
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import eventlet
eventlet.monkey_patch()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DOWNLOAD_FOLDER'] = 'processed'
socketio = SocketIO(app)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'doc'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def update_progress(progress, message):
    print(f'Progress: {progress}%, Message: {message}')
    socketio.emit('progress', {'progress': progress,
                  'message': message}, namespace='/')


@socketio.on('form_submitted', namespace='/')
def form_submitted(data):
    print('Form submitted:', data)


@app.before_first_request
def create_folders():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)


def generate_filename(original_filename: str, extension: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    return f"{timestamp}_{secure_filename(original_filename)}.{extension}"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('JSONL')
        use_case = request.form['use_case']

        if file and allowed_file(file.filename):
            file_size = len(file.read())
            if file_size <= MAX_FILE_SIZE:
                file.seek(0)
                filename = generate_filename(file.filename, 'txt')
                original_file = os.path.join(
                    app.root_path, app.config['UPLOAD_FOLDER'], filename)
                file.save(original_file)
                update_progress(25, 'File uploaded successfully.')

                try:
                    converted_file = os.path.join(
                        app.root_path, app.config['UPLOAD_FOLDER'], os.path.splitext(filename)[0] + ".txt")
                    text = convert_file_to_text(original_file)
                    save_converted_text_to_file(text, converted_file)
                    update_progress(50, 'File converted to text.')

                    if use_case:
                        jsonl_filename = generate_filename(
                            os.path.splitext(filename)[0], 'jsonl')
                        processed_file = os.path.join(
                            app.root_path, app.config['DOWNLOAD_FOLDER'], jsonl_filename)

                        if use_case == 'text_classification':
                            output_file_path = text_classification_pipeline(
                                app.root_path, text)
                        elif use_case == 'sentiment_analysis':
                            output_file_path = sentiment_analysis_pipeline(
                                app.root_path, text)
                        elif use_case == 'entity_extraction':
                            output_file_path = entity_extraction_pipeline(
                                app.root_path, text)
                        elif use_case == 'chatbot':
                            output_file_path = chatbot_pipeline(
                                app.root_path, text)
                        elif use_case == 'product_description':
                            output_file_path = product_description_pipeline(
                                app.root_path, text)

                        os.rename(output_file_path, processed_file)
                        return redirect(url_for('completed', filename=jsonl_filename))

                except Exception as e:
                    tb_str = traceback.format_exception(
                    etype=type(e), value=e, tb=e.__traceback__)
                    print("".join(tb_str))
                    error_message = 'Error processing file: ' + str(e)
                    return render_template('index.html', error_message=error_message)

            else:
                return render_template('index.html', error_message=f'File size too large. Maximum file size allowed is {MAX_FILE_SIZE // (1024 * 1024)} MB.')
        else:
            return render_template('index.html', error_message='Invalid file or file size.')
    else:
        return render_template('index.html', error_message=None)



@app.route('/completed', methods=['GET'])
def completed():
    filename = request.args.get('filename')
    if filename:
        return render_template('completed.html', download_file=filename)
    else:
        return redirect(url_for('index'))


@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    return send_from_directory(directory=app.config['DOWNLOAD_FOLDER'], filename=filename, as_attachment=True)


if __name__ == '__main__':
    socketio.run(app)
