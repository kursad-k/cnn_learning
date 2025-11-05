import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.mobilenet_v2()
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load('image_model.pth', weights_only=True))
model = model.to(device)
model.eval()

img = Image.open(sys.argv[1]).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)
    prob = torch.softmax(output, dim=1)
    pred = output.argmax(1).item()

classes = ['bad', 'good']
print(f'Prediction: {classes[pred]} ({prob[0][pred].item()*100:.1f}%)')
