import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import kagglehub

# 1. SET ENVIRONMENT
os.environ["KAGGLEHUB_CACHE"] = "/content/kaggle_cache"

def train_and_evaluate():
    # 2. DOWNLOAD DATASET
    print("Fetching Kaggle Skin Disease Dataset...")
    raw_path = kagglehub.dataset_download("ismailpromus/skin-diseases-image-dataset")

    # Path correction for nested folders
    dataset_path = raw_path
    for root, dirs, files in os.walk(raw_path):
        if len(dirs) > 5:
            dataset_path = root
            break

    # 3. PREPROCESSING
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    classes = full_dataset.classes
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=2)

    # 4. MODEL SETUP
    model = models.mobilenet_v3_small(weights='DEFAULT')
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 5. TRAINING LOOP WITH PROGRESSION
    print(f"\n🚀 Training on {device} for {len(classes)} classes...")
    for epoch in range(15):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Show progress every 50 batches
            if i % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        acc = 100. * correct / total
        print(f"✅ Epoch {epoch+1} Summary: Loss: {running_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

    # 6. SAVE WEIGHTS
    torch.save(model.state_dict(), 'skin_model.pth')
    print("\n💾 Model saved as skin_model.pth")

    # 7. GENERATE EVALUATION MATRIX (Confusion Matrix)
    print("\n📊 Generating Evaluation Matrix...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Skin Disease Classification - Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Download results
    from google.colab import files
    files.download('skin_model.pth')
    files.download('confusion_matrix.png')

if __name__ == "__main__":
    train_and_evaluate()