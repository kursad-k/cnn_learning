import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.mobilenet_v2(weights='DEFAULT')
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load('image_model.pth', weights_only=True))
model = model.to(device)
model.eval()

folder = Path(sys.argv[1])
img_files = list(folder.glob('*'))
img_files = [f for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

if not img_files:
    print(f'No images found in {folder}')
else:
    for img_path in img_files:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)
            pred = output.argmax(1).item()
        
        result = 'GOOD' if pred == 1 else 'BAD'
        confidence = prob[0][pred].item() * 100
        print(f'{img_path.name}: {result} ({confidence:.1f}%)')
    
    good_count = sum(1 for f in img_files if True)  # Will be calculated in loop
    print(f'\nProcessed {len(img_files)} images')
