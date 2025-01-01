# Face-Centered-Video-Creator

A Flask-based web application that creates face-centered videos from a series of images. The application uses MTCNN for face detection and OpenCV for video processing, ensuring that detected faces remain centered throughout the generated video.

## Features

- Face detection using MTCNN (Multi-task Cascaded Convolutional Networks)
- Multiple face support with face selection capability
- Automatic face centering in output video
- Adjustable video speed
- Web-based interface for easy interaction
- Preview of detected faces before video creation

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd face-centered-video-creator
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── app.py              # Main Flask application
├── templates/          # HTML templates
│   └── index.html     # Main web interface
├── static/            # Static files
│   └── output/        # Generated face previews and videos
├── input_images/      # Directory for input images
└── requirements.txt   # Python dependencies
```

## Usage

1. Place your images in the `input_images` folder.

2. Start the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5001
```

4. Click the "Process Images" button on the web interface.

5. The application will:
   - Detect faces in each image
   - Show previews of detected faces
   - Allow you to select which face to center on (if multiple faces are detected)
   - Generate a video with the selected face centered in each frame

## File Types and Storage

- Supported input image formats: JPG, JPEG, PNG
- Output video format: MP4
- The application automatically manages files in:
  - `input_images/*.jpg` - Input images
  - `static/output/*.jpg` - Face preview images
  - `static/output/*.mp4` - Generated videos

## Performance Optimization

The application includes several optimizations:
- Image resizing before face detection for faster processing
- Efficient numpy operations for image manipulation
- Optimized video encoding with multiple codec support
- Memory-efficient batch processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This project was developed with the assistance of Claude Sonnet 3.5, an AI model by Anthropic.
