import urllib.request
import json


def show_top_rank_5(output):
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    with urllib.request.urlopen(url) as f:
        class_labels_1k = json.load(f)
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
