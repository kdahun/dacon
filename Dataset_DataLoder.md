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

# 2.파일에서 사용자 정의 데이터셋 만들기
사용자 정의 Dataset 클래스는 반드시 3개 함수를 구현해야 한다.
__init__, __len__, __getitem__
  import os
  import pandas as pd
  from torchvision.io import read_image
  
  class CustomImageDataset(Dataset):
      def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
          self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
          self.img_dir = img_dir
          self.transform = transform
          self.target_transform = target_transform
  
      def __len__(self):
          return len(self.img_labels)
  
      def __getitem__(self, idx):
          img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
          image = read_image(img_path)
          label = self.img_labels.iloc[idx, 1]
          if self.transform:
              image = self.transform(image)
          if self.target_transform:
              label = self.target_transform(label)
          return image, label

## __init__
  __init__함수는 Dataset 객체가 생성될때 한 번만 실행된다. 여기서는 이미지와 주석 파일이 포함된 디렉토리와 두가지 변형을 초기화한다.

## __len__
  함수의 데이터셋의 샘플 개수를 반환한다.

## __getitem__
  이 함수는 주어진 인덱스에 해당하는 샘플을 데이터셋에서 불러오고 반환한다. 인덱스를 기반으로, 디스크에서 이미지의 위치를 식별하고, read_image를 사용하여 이미지를 텐서로 변환하고, self.img_labels의 csv 데이터로부터 해당하는 정답을 가져오고, (해당하는 경우_ 변형 함수들을 호출한 뒤, 텐서 이미지와 라벨을 Python 사전형으로 반환한다.
