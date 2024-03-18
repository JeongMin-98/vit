import timm
import torch
import torchsummary as summary
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import urllib.request
import json

model = timm.create_model('vit_base_patch16_224', pretrained=True)

model.cuda()

summary.summary(model, input_size=(3, 224, 224))

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

x = torch.randn(1, 3, 224, 224)
x = x.to('cuda')

pred = model(x)
print(pred.shape)

# x = x.to('cpu')

image_path = 'cello.jpg'
img = Image.open(image_path)

preprocessed_img = transforms(img).to(dtype=torch.float32)
batch = preprocessed_img.unsqueeze(0)

print(batch.shape)

with torch.no_grad():
    batch_tensor = batch.to('cuda')
    output = model(batch_tensor)

print(output.shape)

# ImageNet 클래스 레이블을 다운로드
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
with urllib.request.urlopen(url) as f:
    class_labels_1k = json.load(f)

# rank5
prob5, pred5 = output.topk(5, 1, True, True)
print(pred5.shape)
print(prob5.shape)
pred5 = pred5.t()  # transpose
print(pred5.shape)
for i in range(5):
    print(f'{i + 1}번째 =========================================')
    print(f'{i + 1}번째 높은 확률을 갖는 클래스의 인덱스:', pred5[i].item())
    print(f'{i + 1}번째 높은 확률을 갖는 클래스:', class_labels_1k[pred5[i].item()])
    print(f'{i + 1}번째 높은 확률을 갖는 클래스의 확률:', prob5.t()[i].item())

torch.cuda.empty_cache()
model.to('cpu')
