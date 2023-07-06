# 파이토치 기본 익히기
대부분의 머신러닝 워크플로우는 데이터 작업과 모델 생성, 모델 매개별수 최적화, 학습된 모델 저장이 포함된다.

파이토치에는 데이터 작업을 위한 기본 요소 두가지인 torch.utils.data.DataLoader와 torch.utils.data.Dataset이 있다.
Dataset은 샘플과 정답을 저장하고, DataLoader는 Dataset을 순회 가능한 객체로 감싼다.

    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import ToTensor

PyTorch는 TorchText, TorchVision 및 TorchAudio와 같이 도메인 특화 라이브러리를 데이터셋과 함께 제공한다.
torchvision.datasets 모듈은 CIFAR, COCO 등과 같은 다양한 실제 비전 데이터에 대한 Dataset을 포함하고 있다.

# 공개 데이터셋에서 학습 데이터를 내려받습니다.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

# 공개 데이터셋에서 테스트 데이터를 내려받습니다.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

Dataset을 DataLoader의 인자로 전달한다. 이는 데이터셋을 순회 가능한 객체(iterable)로 감싸고, 자동화된 배치(batch), 샘플링(sampling), 섞기(shuffle) 및 다중 프로세스로 데이터 불러오기(multiprocessdata loading)를 지원한다.
여기에서는 배치 크기를 64로 정의한다. 즉 데이터로더(dataloader) 객체의  각 요소는 64개의 특징(feature)과 정답(label)을 묶음(batch)으로 변환한다.

# 데이터로더를 생성한다.

    batch_size =64

    train_dataloader = DataLoader(training_data,batch_size=batch_size)
    test_dataloader = DataLoader(test_data,batch_size=batch_size)

    for X,y in test_dataloader:
        print(f"Shape of X[N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
