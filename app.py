import os
import traceback
from flask import Flask, request, render_template, jsonify, send_file
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
from werkzeug.serving import WSGIRequestHandler
from werkzeug.exceptions import HTTPException

# Configure Werkzeug to handle broken pipe errors gracefully
WSGIRequestHandler.protocol_version = "HTTP/1.1"

app = Flask(__name__)

# Initialize face detector
from mtcnn import MTCNN
detector = MTCNN()

# Global variable to store processing progress
processing_progress = {'status': '', 'progress': 0, 'total': 0}

# Ensure directories exist with proper permissions
os.makedirs('input_images', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/output', exist_ok=True)

# Clean up old output files
for file in os.listdir('static/output'):
    try:
        os.remove(os.path.join('static/output', file))
    except Exception as e:
        print(f"Error removing old file {file}: {e}")

def update_progress(status, current, total):
    global processing_progress
    processing_progress = {
        'status': status,
        'progress': current,
        'total': total
    }

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Error occurred: {str(e)}")
    print(traceback.format_exc())
    
    if isinstance(e, HTTPException):
        return jsonify({'error': str(e)}), e.code
    
    # Handle broken pipe error
    if isinstance(e, OSError) and e.errno == 32:
        print("Broken pipe error - client disconnected")
        return jsonify({'error': 'Connection lost'}), 500
    
    return jsonify({'error': 'An internal error occurred'}), 500

def detect_faces(image_path):
    try:
        # Read the image
        print(f"Reading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None, []
        
        # Resize image for faster detection
        max_dimension = 1024
        height, width = image.shape[:2]
        scale = min(max_dimension / width, max_dimension / height)
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_image = cv2.resize(image, (new_width, new_height))
        else:
            resized_image = image
            scale = 1
        
        # Detect faces using MTCNN
        faces_info = detector.detect_faces(resized_image)
        
        # Filter faces based on confidence
        confidence_threshold = 0.8
        min_face_size = 60
        
        filtered_faces = []
        for face in faces_info:
            confidence = face['confidence']
            
            if confidence > confidence_threshold:
                # Scale back the coordinates to original image size
                x, y, w, h = face['box']
                x = int(x / scale)
                y = int(y / scale)
                w = int(w / scale)
                h = int(h / scale)
                
                if w >= min_face_size and h >= min_face_size:
                    try:
                        margin = int(min(w, h) * 0.2)
                        x = max(0, x - margin)
                        y = max(0, y - margin)
                        w = min(width - x, w + 2*margin)
                        h = min(height - y, h + 2*margin)
                        
                        # Use numpy slicing for faster face extraction
                        face_img = image[y:y+h, x:x+w]
                        preview_path = f'static/output/face_{os.path.basename(image_path)}_{len(filtered_faces)}.jpg'
                        cv2.imwrite(preview_path, face_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        
                        filtered_faces.append({
                            'id': len(filtered_faces),
                            'location': (x, y, w, h),
                            'preview': preview_path,
                            'confidence': float(confidence)
                        })
                        print(f"Found face with confidence: {confidence:.2f}")
                    except Exception as e:
                        print(f"Error saving face preview: {str(e)}")
                        continue
        
        return image, filtered_faces
    except Exception as e:
        print(f"Error in detect_faces for {image_path}: {str(e)}")
        print(traceback.format_exc())
        return None, []

def create_centered_video(image_paths, selected_faces=None, video_speed=1.0):
    try:
        if not image_paths:
            print("No image paths provided")
            return None
        
        target_size = (640, 480)
        target_face_size = (150, 150)
        target_face_position = (target_size[0]//2 - target_face_size[0]//2,
                              target_size[1]//2 - target_face_size[1]//2)
        
        # Pre-allocate output frames array
        output_frames = []
        
        total_images = len(image_paths)
        for idx, image_path in enumerate(image_paths):
            try:
                update_progress(f"Processing image {idx + 1}/{total_images}", idx + 1, total_images)
                
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to read image: {image_path}")
                    continue
                
                # Detect faces using MTCNN with resized image for speed
                max_dimension = 1024
                height, width = image.shape[:2]
                scale = min(max_dimension / width, max_dimension / height)
                if scale < 1:
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    resized_detect = cv2.resize(image, (new_width, new_height))
                else:
                    resized_detect = image
                    scale = 1
                
                faces_info = detector.detect_faces(resized_detect)
                
                if len(faces_info) == 0:
                    print(f"No faces detected in image: {image_path}")
                    continue
                
                face_idx = selected_faces.get(os.path.basename(image_path), 0) if selected_faces else 0
                if face_idx >= len(faces_info):
                    print(f"Selected face index {face_idx} out of range for image {image_path}")
                    face_idx = 0
                
                # Scale back coordinates
                x, y, w, h = [int(coord / scale) for coord in faces_info[face_idx]['box']]
                
                # Calculate scaling factor
                scale = min(target_face_size[0]/w, target_face_size[1]/h)
                
                # Resize image once
                new_width = int(image.shape[1] * scale)
                new_height = int(image.shape[0] * scale)
                resized_image = cv2.resize(image, (new_width, new_height))
                
                # Create canvas using numpy
                canvas = np.full((target_size[1], target_size[0], 3), 255, dtype=np.uint8)
                
                # Calculate positions
                new_face_x = int(x * scale)
                new_face_y = int(y * scale)
                paste_x = target_face_position[0] - new_face_x
                paste_y = target_face_position[1] - new_face_y
                
                # Use numpy's advanced indexing for faster image composition
                x_start = max(0, paste_x)
                y_start = max(0, paste_y)
                x_end = min(target_size[0], paste_x + new_width)
                y_end = min(target_size[1], paste_y + new_height)
                img_x_start = max(0, -paste_x)
                img_y_start = max(0, -paste_y)
                img_x_end = img_x_start + (x_end - x_start)
                img_y_end = img_y_start + (y_end - y_start)
                
                canvas[y_start:y_end, x_start:x_end] = resized_image[img_y_start:img_y_end, img_x_start:img_x_end]
                output_frames.append(canvas)
                
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                print(traceback.format_exc())
                continue
        
        if not output_frames:
            print("No frames generated for video")
            return None
        
        try:
            update_progress("Creating video", total_images, total_images)
            output_path = 'static/output/output_video.mp4'
            codecs = ['avc1', 'mp4v', 'XVID']
            base_fps = 24.0
            adjusted_fps = base_fps * video_speed
            
            for codec in codecs:
                try:
                    print(f"Trying codec: {codec}")
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(output_path, fourcc, adjusted_fps, target_size)
                    
                    # Write frames in batch
                    for frame in output_frames:
                        out.write(frame)
                    
                    out.release()
                    success = True
                    break
                except Exception as e:
                    print(f"Failed with codec {codec}: {str(e)}")
                    continue
            
            if not success:
                print("Failed to create video with any codec")
                return None
            
            return output_path
            
        except Exception as e:
            print(f"Error creating video: {str(e)}")
            print(traceback.format_exc())
            return None
            
    except Exception as e:
        print(f"Error in create_centered_video: {str(e)}")
        print(traceback.format_exc())
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_images', methods=['POST'])
def process_images():
    try:
        print("Starting process_images")
        image_files = sorted(os.listdir('input_images'))  # Sort image files
        print(f"Found {len(image_files)} files in input_images directory")
        
        image_paths = [os.path.join('input_images', f) for f in image_files 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(image_paths)} image files")
        
        if not image_paths:
            print("No images found in input_images directory")
            return jsonify({'error': 'No images found in input_images directory'})
        
        # Reset progress
        update_progress("Processing images", 0, len(image_paths))
        
        # Detect faces in all images
        faces_data = {}
        skipped_images = []
        
        for idx, image_path in enumerate(image_paths):
            print(f"\nProcessing image {idx + 1}/{len(image_paths)}: {image_path}")
            update_progress(f"Detecting faces in image {idx + 1}/{len(image_paths)}", idx + 1, len(image_paths))
            
            image, faces_info = detect_faces(image_path)
            
            if faces_info:
                print(f"Found {len(faces_info)} faces in {image_path}")
                faces_data[os.path.basename(image_path)] = faces_info
            else:
                print(f"No faces found in {image_path}")
                skipped_images.append(os.path.basename(image_path))
        
        print(f"\nProcessing complete:")
        print(f"- Images with faces: {len(faces_data)}")
        print(f"- Skipped images: {len(skipped_images)}")
        
        if not faces_data:
            print("No faces detected in any images")
            return jsonify({'error': 'No faces detected in any images'})
        
        # Return both faces data and skipped images
        result = {
            'status': 'selection_needed',
            'faces_data': faces_data,
            'skipped_images': skipped_images
        }
        
        # Debug the JSON serialization
        import json
        try:
            json_str = json.dumps(result)
            print("Successfully serialized result to JSON")
            return jsonify(result)
        except TypeError as e:
            print(f"JSON serialization error: {str(e)}")
            print("Result structure:")
            print(f"faces_data keys: {list(faces_data.keys())}")
            for img, faces in faces_data.items():
                print(f"Image {img} faces: {faces}")
            return jsonify({'error': 'Error processing image data'})
        
    except Exception as e:
        print(f"Error in process_images: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'An error occurred: {str(e)}'})

@app.route('/create_video', methods=['POST'])
def create_video():
    try:
        data = request.json
        selected_faces = data.get('selected_faces', {})
        video_speed = float(data.get('video_speed', 1.0))
        
        image_files = os.listdir('input_images')
        # Only include selected images that have faces and sort them by filename
        image_paths = sorted([os.path.join('input_images', f) for f in image_files 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f in selected_faces])
        
        if not image_paths:
            return jsonify({'error': 'No images selected for video creation'})
        
        print("Processing images in order:")
        for path in image_paths:
            print(f"- {os.path.basename(path)}")
        
        video_path = create_centered_video(image_paths, selected_faces, video_speed)
        if video_path:
            return jsonify({
                'status': 'success',
                'video_url': video_path
            })
        
        return jsonify({'error': 'Failed to create video. Check server logs for details.'})
        
    except Exception as e:
        print(f"Error in create_video: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'An error occurred: {str(e)}'})

@app.route('/progress')
def get_progress():
    return jsonify(processing_progress)

if __name__ == '__main__':
    # Use threaded=True for better handling of multiple requests
    app.run(debug=True, port=5001, threaded=True)
