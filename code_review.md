    class SatelliteDataset(Dataset):
        def __init__(self, csv_file, transform=None, infer=False):
            self.data = pd.read_csv(csv_file)
            self.transform = transform
            self.infer = infer

        def __len__(self): # 데이터의 길이를 반환해 주는 듯
            return len(self.data)

        def __getitem__(self, idx):
            img_path = self.data.iloc[idx, 1]  # 
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
            if self.infer:
                if self.transform:
                    image = self.transform(image=image)['image']
                return image

            mask_rle = self.data.iloc[idx, 2]
            mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            return image, mask

+ __init__ : 이 메서드는 클래스의 인스턴스가 생성될 때 자동으로 호출되는 생성자 메서드
+ __len__ : len() 함수가 클래스의 객체에 대해 호출될 때 자동으로 호출되는 메서드로, 객체의 길이 또는 크기를 반환한다. 데이터셋의 전체 길이를 반환하는 역할을 한다. len() 함수에 클래스의 객체를 전달하면 __len__ 메서드가 홀출되어 객체의 길이를 반환한다.
+ __getitem__ : 인덱싱을 통해 클래스의 객체의  요소에 접근할 때 자동으로 호출되는 메서드이다. [] 연산자를 사용하여 객체의 요소에 접근할 때 호출된다. __getitem__메서드는 인덱스에 해당하는 요소를 반환한다. 데이터셋 객체를 인덱싱하면 __getitme__ 메서드가 호출되어 해당 인덱스에 대한 데이터를 반환한다.
