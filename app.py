from flask import Flask, render_template, request, redirect, send_file
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image as PILImage

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def apply_filter(image, filter_type, params=None):
    if filter_type == 'average':
        return cv2.blur(image, (3, 3))
    elif filter_type == 'median':
        return cv2.medianBlur(image, 3)
    elif filter_type == 'edge':
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        # Convert back to 3-channel for consistent histogram calculation
        if len(image.shape) == 3:
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return edges
    elif filter_type == 'sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == 'smooth':
        kernel = np.ones((5,5), np.float32)/25
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == 'rotate':
        angle = int(params.get('angle', 90))
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        return cv2.warpAffine(image, rotation_matrix, (width, height))
    elif filter_type == 'resize':
        scale = float(params.get('scale', 1.5))
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale), int(width * scale)
        return cv2.resize(image, (new_width, new_height))
    elif filter_type == 'mirror_horizontal':
        return cv2.flip(image, 1)  # 1 for horizontal flip
    elif filter_type == 'mirror_vertical':
        return cv2.flip(image, 0)  # 0 for vertical flip
    elif filter_type == 'center_of_mass':
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Threshold the image to create binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Calculate moments
        M = cv2.moments(binary)
        
        # Calculate center of mass (centroid)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            # If no valid moments (e.g., all black image), use center of image
            cX, cY = gray.shape[1] // 2, gray.shape[0] // 2
        
        # Create a copy of the original image
        result = image.copy() if len(image.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Draw a red circle at the center of mass
        cv2.circle(result, (cX, cY), 10, (0, 0, 255), -1)
        
        # Draw crosshair lines
        cv2.line(result, (cX - 20, cY), (cX + 20, cY), (0, 0, 255), 2)
        cv2.line(result, (cX, cY - 20), (cX, cY + 20), (0, 0, 255), 2)
        
        # Add text with coordinates
        cv2.putText(result, f"({cX}, {cY})", (cX + 20, cY + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result
        
    elif filter_type == 'skeletonize':
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Threshold the image to create binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Create an empty skeleton image
        skeleton = np.zeros(binary.shape, np.uint8)
        
        # Get a cross-shaped structuring element
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        # Copy the binary image
        img = binary.copy()
        
        # Skeletonization algorithm
        while True:
            # Step 1: Open the image
            open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            # Step 2: Subtract open from the original
            temp = cv2.subtract(img, open_img)
            # Step 3: Erode the original image
            eroded = cv2.erode(img, element)
            # Step 4: Get the skeleton parts
            skeleton = cv2.bitwise_or(skeleton, temp)
            # Step 5: Set the eroded image for next iteration
            img = eroded.copy()
            
            # If there are no white pixels left, we're done
            if cv2.countNonZero(img) == 0:
                break
        
        # Convert back to 3-channel if original was color
        if len(image.shape) == 3:
            skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        
        return skeleton
    
    elif filter_type == 'hist_equalize':
        if len(image.shape) == 3:  # Color image
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        else:  # Grayscale image
            return cv2.equalizeHist(image)
    elif filter_type == 'contrast_stretch':
        min_val = int(params.get('min', 50))
        max_val = int(params.get('max', 200))
        # Convert to appropriate range
        min_percent = min_val / 255.0
        max_percent = max_val / 255.0
        
        # Apply contrast stretching
        min_val = np.percentile(image, min_percent * 100)
        max_val = np.percentile(image, max_percent * 100)
        
        # Clip image
        stretched = np.clip((image - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)
        return stretched
    # Manual thresholding
    elif filter_type == 'threshold_manual':
        thresh_value = int(params.get('thresh_value', 128))
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # Apply thresholding
        _, binary = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
        # Convert back to 3-channel if original was color
        if len(image.shape) == 3:
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return binary
    # Otsu thresholding
    elif filter_type == 'threshold_otsu':
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Convert back to 3-channel if original was color
        if len(image.shape) == 3:
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return binary
    # Kapur thresholding (entropy-based)
    elif filter_type == 'threshold_kapur':
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate histogram and normalize it
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Calculate threshold using Kapur's entropy method
        threshold = 0
        max_entropy = 0
        
        for t in range(1, 255):
            # Background
            bg_hist = hist[:t]
            bg_sum = bg_hist.sum()
            
            # Foreground
            fg_hist = hist[t:]
            fg_sum = fg_hist.sum()
            
            # Avoid division by zero
            if bg_sum > 0 and fg_sum > 0:
                # Normalize histograms
                bg_hist_norm = bg_hist / bg_sum
                fg_hist_norm = fg_hist / fg_sum
                
                # Calculate entropies (avoiding log(0))
                bg_entropy = -np.sum(bg_hist_norm * np.log2(bg_hist_norm + 1e-10))
                fg_entropy = -np.sum(fg_hist_norm * np.log2(fg_hist_norm + 1e-10))
                
                # Total entropy
                total_entropy = bg_entropy + fg_entropy
                
                # Update threshold if entropy is higher
                if total_entropy > max_entropy:
                    max_entropy = total_entropy
                    threshold = t
        
        # Apply threshold
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Convert back to 3-channel if original was color
        if len(image.shape) == 3:
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return binary
    # Dilation operation (morphological)
    elif filter_type == 'dilation':
        # Get kernel size from params or use default
        kernel_size = int(params.get('kernel_size', 5))
        # Create kernel for dilation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # If color image, convert to binary first
        if len(image.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply threshold to get binary
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            # Apply dilation
            dilated = cv2.dilate(binary, kernel, iterations=1)
            # Convert back to BGR
            return cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        else:
            # If already grayscale, apply threshold and dilation
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return cv2.dilate(binary, kernel, iterations=1)
    # Erosion operation (morphological)
    elif filter_type == 'erosion':
        # Get kernel size from params or use default
        kernel_size = int(params.get('kernel_size', 5))
        # Create kernel for erosion
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # If color image, convert to binary first
        if len(image.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply threshold to get binary
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            # Apply erosion
            eroded = cv2.erode(binary, kernel, iterations=1)
            # Convert back to BGR
            return cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
        else:
            # If already grayscale, apply threshold and erosion
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return cv2.erode(binary, kernel, iterations=1)
    else:
        return image

def calculate_histogram(image):
    hist_img = np.zeros((300, 256, 3), dtype=np.uint8)
    
    # Check if the image is grayscale or color
    if len(image.shape) < 3 or image.shape[2] == 1:
        # Grayscale image - plot in blue
        histr = cv2.calcHist([image], [0], None, [256], [0, 256])
        cv2.normalize(histr, histr, 0, 300, cv2.NORM_MINMAX)
        for j in range(1, 256):
            cv2.line(hist_img, (j-1, 300-int(histr[j-1])), (j, 300-int(histr[j])),
                    (255, 0, 0), 1)  # Blue for grayscale
    else:
        # Color image
        color = ('b','g','r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([image], [i], None, [256], [0, 256])
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

        # Eğer dosya .tiff veya .tif ise PNG'ye dönüştür
        if filename.lower().endswith(('.tif', '.tiff')):
            try:
                new_filename = filename.rsplit('.', 1)[0] + '.png'
                new_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
                with PILImage.open(filepath) as img:
                    img.convert("RGB").save(new_path, "PNG")
                filename = new_filename  # HTML tarafı artık .png görecek
            except Exception as e:
                return f"TIFF dosyası dönüştürülürken hata oluştu: {str(e)}", 500

        return redirect(f'/edit/{filename}')
    return render_template('index.html')

@app.route('/edit/<filename>', methods=['GET', 'POST'])
def edit(filename):
    # .tiff ismi geldiyse ama .png varsa onu kullan
    if filename.lower().endswith(('.tif', '.tiff')):
        png_filename = filename.rsplit('.', 1)[0] + '.png'
        png_path = os.path.join(app.config['UPLOAD_FOLDER'], png_filename)
        if os.path.exists(png_path):
            filename = png_filename

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(image_path)

    if image is None:
        try:
            pil_image = PILImage.open(image_path).convert('RGB')
            image = np.array(pil_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            return f"Görüntü yüklenemedi! Hata: {str(e)}", 400

    processed_image = image
    hist_img = None
    error_msg = None
    action = None

    try:
        original_hist_img = calculate_histogram(image)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f'orig_hist_{filename}'), original_hist_img)
    except Exception as e:
        print(f"Original histogram error: {str(e)}")
        original_hist_img = np.zeros((300, 256, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f'orig_hist_{filename}'), original_hist_img)

    if request.method == 'POST':
        action = request.form.get('action')
        params = {}
        if action == 'rotate':
            params['angle'] = request.form.get('angle', '90')
        elif action == 'resize':
            params['scale'] = request.form.get('scale', '1.5')
        elif action == 'contrast_stretch':
            params['min'] = request.form.get('min', '50')
            params['max'] = request.form.get('max', '200')
        elif action == 'threshold_manual':
            params['thresh_value'] = request.form.get('thresh_value', '128')
        elif action == 'dilation' or action == 'erosion':
            params['kernel_size'] = request.form.get('kernel_size', '5')

        try:
            processed_image = apply_filter(image, action, params)
            if processed_image is None or processed_image.size == 0:
                error_msg = "İşlenmiş görüntü oluşturulamadı!"
                processed_image = image
            else:
                try:
                    hist_img = calculate_histogram(processed_image)
                    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f'hist_{filename}'), hist_img)
                except Exception as e:
                    print(f"Histogram calculation error: {str(e)}")
                    hist_img = np.zeros((300, 256, 3), dtype=np.uint8)
                    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f'hist_{filename}'), hist_img)

                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}'), processed_image)
        except Exception as e:
            error_msg = f"İşlem sırasında bir hata oluştu: {str(e)}"
            print(f"Error in image processing: {str(e)}")

    has_processed = os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}'))
    return render_template('edit.html',
                       filename=filename,
                       has_processed=has_processed,
                       error_msg=error_msg,
                       selected_action=action)

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)