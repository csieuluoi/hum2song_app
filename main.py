# from flask import Flask, request, jsonify, flash, redirect, render_template

# import uuid

# import os
# import sys
# import inspect
# from datetime import datetime

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir) 
# sys.path.insert(0, currentdir) 

# from torch_utils import get_result, load_dependencies, CFG



# UPLOAD_FOLDER = 'files'
# FILE_NAME = "hum.wav"

# app = Flask(__name__)

# model, cfg, index2id = load_dependencies(root_path="./checkpoints", checkpoint_name="resnet18_latest.pth")

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ALLOWED_EXTENSIONS = {'npy', 'mp3', 'wav'}
# def allowed_file(filename):
#     # xxx.png
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # get thumnail of yt video


# @app.route('/')
# def root():
#     return app.send_static_file('index.html')


# @app.route('/predict', methods=["GET"])
# def predict():
#     if request.method == 'GET':
#         root_hum = UPLOAD_FOLDER
#         if FILE_NAME == "":
#             return render_template('error.html', error='No file')
#         if not allowed_file(FILE_NAME):
#             return render_template('error.html', error=f'Format not supported: {FILE_NAME}')

#         prediction = get_result(root_hum, FILE_NAME, model, cfg, index2id)
#         # prediction = get_result(root_hum, "0283.mp3", model, cfg, index2id)


#         data = {'prediction': prediction}
        
#         return render_template('prediction.html', data=data)


# @app.route('/save-record', methods=['POST'])
# def save_record():
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)

#     file = request.files['file']

#     if file.filename == '':
#         flash('No selected file')
#         return redirect(request.url)

#     if not allowed_file(file.filename):
#         flash(f'Format not supported: {file.filename}')
#         return redirect(request.url)

#     if not os.path.exists(app.config['UPLOAD_FOLDER']):
#         os.mkdir(app.config['UPLOAD_FOLDER'])

    
#     full_file_path = os.path.join(app.config['UPLOAD_FOLDER'], FILE_NAME)
#     file.save(full_file_path)

#     return jsonify({'success': True, 'filename': FILE_NAME})

# if __name__ == "__main__":

#     app.run(debug=True)


from flask import Flask, request, jsonify, flash, redirect, render_template
from googleapiclient.discovery import build
import uuid
import os
import sys
import inspect
from datetime import datetime

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)

from torch_utils import get_result, load_dependencies, CFG


UPLOAD_FOLDER = 'files'
FILE_NAME = "hum.wav"

app = Flask(__name__)

model, cfg, index2id = load_dependencies(root_path="./checkpoints", checkpoint_name="resnet18_latest.pth")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'npy', 'mp3', 'wav'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/predict', methods=["GET"])
def predict():
    if request.method == 'GET':
        root_hum = UPLOAD_FOLDER
        if FILE_NAME == "":
            return render_template('error.html', error='No file')
        if not allowed_file(FILE_NAME):
            return render_template('error.html', error=f'Format not supported: {FILE_NAME}')

        prediction = get_result(root_hum, FILE_NAME, model, cfg, index2id)

        data = {'prediction': prediction}

        return render_template('prediction.html', data=data)



@app.route('/save-record', methods=['POST'])
def save_record():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash(f'Format not supported: {file.filename}')
        return redirect(request.url)

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.mkdir(app.config['UPLOAD_FOLDER'])

    full_file_path = os.path.join(app.config['UPLOAD_FOLDER'], FILE_NAME)
    file.save(full_file_path)

    return jsonify({'success': True, 'filename': FILE_NAME})


if __name__ == "__main__":
    app.run(debug=True)
