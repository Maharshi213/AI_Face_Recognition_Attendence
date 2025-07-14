import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

import torch.nn as nn
import torch.nn.functional as F

class ImprovedFaceCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ImprovedFaceCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Using padding to maintain spatial dimensions
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for better training stability
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # Assuming 100x100 input images after the three 2x2 pooling layers: 100/2/2/2 = 12.5 → 12
        # So feature map size will be 12x12 with 128 channels
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.dropout1 = nn.Dropout(0.5)  # Dropout to prevent overfitting
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten and feed to fully connected layers
        x = x.view(-1, 128 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x

# Define the class names
CLASS_NAMES = ["Aliefiah", "Apurva", "Dhruhi", "kavya","Maharshi"]
num_classes = len(CLASS_NAMES)

# Define your model path
MODEL_PATH = 'student_face_full_model.pth'

def check_model_file():
    """Check if the model file exists and print its size"""
    if os.path.exists(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # Size in MB
        print(f"✅ Model file exists: {MODEL_PATH}")
        print(f"📊 File size: {size:.2f} MB")
        return True
    else:
        print(f"❌ Error: Model file not found: {MODEL_PATH}")
        print(f"💡 Current working directory: {os.getcwd()}")
        print(f"💡 Files in current directory: {os.listdir('.')}")
        return False

def try_load_model():
    """Try to load the model and print diagnostic information"""
    try:
        print("\n📂 Attempting to load model...")
        model = ImprovedFaceCNN(num_classes=num_classes)
        
        # Try to load the model
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        print("\n📋 Model state_dict keys:")
        for key in state_dict.keys():
            print(f"  - {key}")
        
        print("\n📋 Model architecture parameters:")
        for name, param in model.named_parameters():
            print(f"  - {name}: {param.shape}")
        
        # Check if the keys match
        model_keys = set([name for name, _ in model.named_parameters()])
        state_dict_keys = set(state_dict.keys())
        
        if not model_keys.issubset(state_dict_keys):
            print("\n⚠️ Warning: Some model keys are missing from the state_dict")
            missing_keys = model_keys - state_dict_keys
            print(f"Missing keys: {missing_keys}")
        
        if not state_dict_keys.issubset(model_keys):
            print("\n⚠️ Warning: Some state_dict keys are not in the model")
            extra_keys = state_dict_keys - model_keys
            print(f"Extra keys: {extra_keys}")
        
        # Check the final layer shape to verify num_classes
        if 'fc2.weight' in state_dict:
            fc2_shape = state_dict['fc2.weight'].shape
            loaded_num_classes = fc2_shape[0]
            print(f"\n📊 Number of classes in loaded model: {loaded_num_classes}")
            if loaded_num_classes != num_classes:
                print(f"⚠️ Warning: Number of classes in model ({loaded_num_classes}) does not match expected ({num_classes})")
        
        # Try loading the state_dict into the model
        print("\n🔄 Loading state_dict into model...")
        model.load_state_dict(state_dict)
        print("✅ Model loaded successfully!")
        
        # Set model to evaluation mode
        model.eval()
        print("✅ Model set to evaluation mode")
        
        return model
    
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print(f"Type: {type(e)}")
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        return None

if __name__ == "__main__":
    print("=" * 50)
    print("Model Diagnostic Tool")
    print("=" * 50)
    
    if check_model_file():
        model = try_load_model()
        if model is not None:
            print("\n✅ Model diagnostics passed. The model should work in your Flask app.")
        else:
            print("\n❌ Model diagnostics failed. Please check the errors above.")
    
    print("\n" + "=" * 50)