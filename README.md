# ImageProcessingWebSite
A web-based image processing app built with Flask. Users can upload images, apply filters (mean, median, edge detection), view histograms, and download results. Processing uses OpenCV, with some filters manually implemented.

This project is a web image processing application developed using the Python-based Flask framework. Users can upload an image, apply filters (mean, median, edge detection, etc.), view histogram analysis and download processed images through the web interface. Basic image processing functions will be done with OpenCV and some filtering operations will be coded so that they can be applied directly manually.

# Image Processing Web Application

This project is a web-based image processing application developed with the Flask framework. Users can apply various filters on uploaded images, view histogram analysis and download processed images to their computers.

## 🔧 Features

- Upload images in formats such as `.jpg`, `.png`, `.bmp`
- Average, median, edge detection, sharpening, smoothing filters
- Process and display image and histogram separately
- Right click and download images
- Simple and intuitive web interface (with HTML)

## 🚀 Installation and Operation

### 1. Required Libraries

Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate # or for Windows: venv\Scripts\activate
