import timm
import torch
import torchsummary as summary
import urllib.request
import json
from utils.torch_utils import check_device
from utils.dataLoader import custom_transform, load_images

# Only inference
if __name__ == '__main__':
    model = timm.create_model('vit_base_patch16_224', pretrained=True)

    device = check_device()

    model.to(device)

    summary.summary(model, input_size=(3, 224, 224))

    # Load an image for inference
    img = load_images('cello.jpg')

    # preprocessed_img
    preprocessed_img = custom_transform(img).to(dtype=torch.float32)
    batch = preprocessed_img.unsqueeze(0)

    print(batch.shape)

    with torch.no_grad():
        batch_tensor = batch.to(device)
        output = model(batch_tensor)

    print(output.shape)

    # load list of labels
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    with urllib.request.urlopen(url) as f:
        class_labels_1k = json.load(f)

    # Top rank 5
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

    # After inference
    torch.cuda.empty_cache()
    model.to('cpu')
