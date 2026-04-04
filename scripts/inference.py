import torch
from torchvision import models, transforms
from PIL import Image
import io

class SkinClassifier:
    def __init__(self, model_path, num_classes=10): # Changed from 30 to 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Initialize the architecture
        self.model = models.mobilenet_v3_small()
        
        # 2. Modify the final layer to match the 10 classes in your .pth file
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = torch.nn.Linear(in_features, num_classes)
        
        # 3. Load the weights
        # The weights expect 10 classes, so this will now succeed.
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_t)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, index = torch.max(probabilities, 0)
            
        return index.item(), confidence.item()