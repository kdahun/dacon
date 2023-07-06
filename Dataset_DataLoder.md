# DATASET과 DATALOADER

데이터 샘플을 처리하는 코드는 지저분하고 유지보수가 어려울 수 있다. 더 나은 가독성과  모듈성을 위해 데이터셋 코드를 모델 학습 코드로부터 분리하는 것이 이상적이다. PyTorch는 torch.utils.data.DataLoader와 torch.utils.data.Dataset의 두 가지 데이터 기본 요소를 제공하여 미리 준비해둔 데이터셋 뿐만 아니라 가지고 있는 데이터를 사용할 수 있도록한다. Dataset은 샘플과 정답을 저장하고, DataLoader는 Dataset을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 감쌈니다.

PyTorch의 도메인 특화 라이브러리들을 미리 준비해둔 다양한 데이터셋을 제공한다. 데이터셋은 torch.utils.data.Dataset의 하위 클래스로 개별 데이터를 특정하는 함수가 구현되어 있다. 이러한 데이터셋은 모델을 만들어보고 성능을 측정하는데 사용할 수 있다. 

# 1.데이터셋 불러오기

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMINIST(
  root="data", 
  train=True,
  download=True,
  transform=ToTensor()
)

test_data = datasets.FashionMINIST(
  root = "data",
  train=False,
  download=True,
  transform=ToTensor()
)

+ root는 학습/테스트 데이터가 저장되는 경로이다.
+ train은 학습용 또는 테스트용 데이터셋 여부를 지정한다.
+ download=True는 root에 데이터가 없는 경우 인터넷에서 다운로드한다.
+ transform 과 target_transform은 특징(feature)과 정답(label) 변형(transform)을 지정한다.

