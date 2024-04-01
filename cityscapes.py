import glob
from torch.utils.data import Dataset
from PIL import Image

class CityScapes(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.files = sorted(glob.glob(f"{data_path}/train/*.jpg"))
        print(self.files)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        width, height = img.size
        imgA = img.crop((0, 0, width//2, height))
        imgB = img.crop((width//2, 0, width, height))
        
        if self.transform:
            imgA, imgB = self.transform(imgA, imgB)
        
        return imgA, imgB


# if __name__ == '__main__':
#     data_path = "D:\Datasets\cityscapes"
#     ds = CityScapes(data_path)
#     img_A_0, img_B_0 = ds[0]
#     img_A_0.show()
#     img_B_0.show()