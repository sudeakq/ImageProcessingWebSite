from flask import Flask, render_template, request, redirect, send_file
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def apply_filter(image, filter_type):
    if filter_type == 'average':
        return cv2.blur(image, (3, 3))
    elif filter_type == 'median':
        return cv2.medianBlur(image, 3)
    elif filter_type == 'edge':
        return cv2.Canny(image, 100, 200)
    elif filter_type == 'sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == 'smooth':
        kernel = np.ones((5,5), np.float32)/25
        return cv2.filter2D(image, -1, kernel)
    else:
        return image

def calculate_histogram(image):
    color = ('b','g','r')
    hist_img = np.zeros((300, 256, 3), dtype=np.uint8)
    for i, col in enumerate(color):
        histr = cv2.calcHist([image],[i],None,[256],[0,256])
        cv2.normalize(histr, histr, 0, 300, cv2.NORM_MINMAX)
        for j in range(1, 256):
            cv2.line(hist_img, (j-1, 300-int(histr[j-1])), (j, 300-int(histr[j])),
                     (255 if col=='b' else 0, 255 if col=='g' else 0, 255 if col=='r' else 0), 1)
    return hist_img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(f'/edit/{filename}')
    return render_template('index.html')

@app.route('/edit/<filename>', methods=['GET', 'POST'])
def edit(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(image_path)
    processed_image = image
    hist_img = None
    if request.method == 'POST':
        action = request.form.get('action')
        processed_image = apply_filter(image, action)
        hist_img = calculate_histogram(processed_image)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}'), processed_image)
        if hist_img is not None:
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f'hist_{filename}'), hist_img)
    return render_template('edit.html', filename=filename)

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
