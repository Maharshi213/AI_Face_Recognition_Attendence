# Smart Attendance System
An AI-powered Face Recognition-Based Attendance System built with Python, TensorFlow, Keras, OpenCV, Flask, and CNN. This project automates the process of taking attendance by recognizing student faces in real time and logging their records automatically.

##Features 
- **Real-time Face Recognition**: Automatically detects and recognizes student faces using webcam.
- **CNN Model**: Custom-trained Convolutional Neural Network for precise face classification
- **Automated Attendance Logging**: Saves attendance records in CSV/Excel format with timestamps
- **User-friendly Web Interface**: Flask-based web application for easy interaction
- **Cloud Training Support**: Integration with Google Colab for model training

##Technologies Used
- **Python**-Core programming language
- **TensorFlow & Keras** - Deep learning framework for CNN model
- **OpenCV** - Computer vision library for face detection and image processing
- **Flask** - Web framework for user interface
- **Pandas** - Data manipulation and CSV/Excel file handling
- **Google Colab** - Cloud-based model training environment

##How It Works

1. Data Collection & Preprocessing
   - Collect student images for training dataset
   -  preprocessing techniques:
   - Resize images to standard dimensions
   - Convert to grayscale
   - Normalize pixel values
   - Data augmentation for better generalization
2. CNN Model Training
   - Custom Convolutional Neural Network architecture
   - Training on Google Colab for faster processing
   - Model validation and performance optimization
   - Save trained model for deployment
3. Real-time Recognition
   - Webcam captures live video feed
   - OpenCV detects faces in real-time
   - Preprocessed faces fed to trained CNN model
   - Student identification and attendance marking
4. Attendance Management
   - Automatic logging of attendance with timestamps
   - Export to CSV/Excel formats using Pandas
   - Web interface for viewing attendance records


