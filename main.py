import timm
import torch
import torchsummary as summary
import urllib.request
import json
from utils.torch_utils import check_device
from utils.dataLoader import custom_transform, load_image, load_dataset
from utils.inferneceTools import show_top_rank_5

# Only inference
if __name__ == '__main__':
    model = timm.create_model('vit_base_patch16_224', pretrained=True)

    device = check_device()

    model.to(device)

    summary.summary(model, input_size=(3, 224, 224))

    # Load an image for inference
    # img = load_image('image/cello.jpg')
    dataset = load_dataset('image/')
    # preprocessed_img
    for img in dataset:
        preprocessed_img = custom_transform(img[0]).to(dtype=torch.float32)
        batch = preprocessed_img.unsqueeze(0)

        print(batch.shape)

        with torch.no_grad():
            batch_tensor = batch.to(device)
            output = model(batch_tensor)

        print(output.shape)

        show_top_rank_5(output)

    # After inference
    torch.cuda.empty_cache()
    model.to('cpu')
