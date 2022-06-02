import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms

dir_name = os.path.dirname(os.path.abspath(__file__))

### loading Rain800 training dataset
class Rain800TrainData(Dataset):
	def __init__(self, image_size, dataset_dir='/Rain-800/'):
		self.img_transforms = self.build_transform(image_size)
		self.image_size = image_size
		if dataset_dir == '/Rain-800/':
			self.dataset_name = 'Rain-800'
			self.dataset_dir = dir_name + dataset_dir + "training/"
			self.dataset_size = 700
			print("Loading Rain800")
		elif dataset_dir == '/MixSyntheticRainDataset/':
			self.dataset_name = 'MixSyntheticRainDataset'
			self.dataset_dir = dir_name + dataset_dir
			self.dataset_size = 13712
			print("Loading Rain13k")
		elif dataset_dir == '/Rain100L-Train/':
			self.dataset_name = 'Rain100L-Train'
			self.dataset_dir = dir_name + dataset_dir
			self.dataset_size = 1800
			print("Loading Rain100L-Train")
		elif dataset_dir == '/Rain100H-Train/':
			self.dataset_name = 'Rain100H-Train'
			self.dataset_dir = dir_name + dataset_dir
			self.dataset_size = 1800
			print("Loading Rain100H-Train")
		elif dataset_dir == '/Snow100K-training/':
			self.dataset_name = 'Snow100K-training'
			self.dataset_dir = dir_name + dataset_dir
			self.dataset_size = 49999
		else:
			raise Exception("Not a valid datatset directory")

	#image transform
	def build_transform(self, image_size):
	    t = []
	    t.append(transforms.ToTensor())#convert (B, H, W, C) from [0,255] to (B, C, H, W) [0. ,1.]
	    return transforms.Compose(t)

	def __len__(self):
		return self.dataset_size

	def __getitem__(self, index):
		#since our name starts at 1 but index starts at 0
		index += 1
		train_data = None
		test_data = None
		H = 0
		W = 0

		if self.dataset_name=='Rain-800':
			#open image with PIL
			img_dir = self.dataset_dir + str(index) + '.jpg'
			img = Image.open(img_dir)

			#find training data and test data
			img_data = np.asarray(img)
			H, W, _ = img_data.shape
			train_data = img_data[:, W//2:, :]
			test_data = img_data[:, :W//2, :]
			H, W, _ = train_data.shape ### update the shape after spliting the input and label

		elif self.dataset_name=='MixSyntheticRainDataset':
			input_dir = self.dataset_dir + 'input/' + str(index) + '.jpg'
			label_dir = self.dataset_dir + 'label/' + str(index) + '.jpg'
			input_img = Image.open(input_dir)
			label_img = Image.open(label_dir)
			train_data = np.asarray(input_img)
			test_data = np.asarray(label_img)
			H, W, _ = train_data.shape

		elif self.dataset_name == 'Rain100L-Train':
			input_dir = self.dataset_dir + 'rain/' + 'norain-' + str(index) + 'x2.png'
			label_dir = self.dataset_dir + 'norain/' + 'norain-' + str(index) + '.png'
			input_img = Image.open(input_dir)
			label_img = Image.open(label_dir)
			train_data = np.asarray(input_img)
			test_data = np.asarray(label_img)
			H, W, _ = train_data.shape

		elif self.dataset_name == 'Rain100H-Train':
			input_dir = self.dataset_dir + 'rain/X2/' + 'norain-' + str(index) + 'x2.png'
			label_dir = self.dataset_dir + 'norain/' + 'norain-' + str(index) + '.png'
			input_img = Image.open(input_dir)
			label_img = Image.open(label_dir)
			train_data = np.asarray(input_img)
			test_data = np.asarray(label_img)
			H, W, _ = train_data.shape

		elif self.dataset_name == 'Snow100K-training':
			input_dir = self.dataset_dir + 'all/synthetic/' + str(index) + '.jpg'
			label_dir = self.dataset_dir + 'all/gt/' + str(index) + '.jpg'
			input_img = Image.open(input_dir)
			label_img = Image.open(label_dir)
			train_data = np.asarray(input_img)
			test_data = np.asarray(label_img)
			H, W, _ = train_data.shape


		else:
			raise Exception("Not a valid dataset directory")
		
		#randomly crop the test and train image with the same region
		hight_left = H - self.image_size
		width_left = W - self.image_size
		r = 0 
		c = 0 
		if hight_left > 0:
			r = np.random.randint(0, hight_left)
		if width_left > 0:
			c = np.random.randint(0, width_left)
		#print(f"Randomly Cropping: r={r}, c={c}")
		train_data = train_data[r:r+self.image_size, c:c+self.image_size, :]
		test_data = test_data[r:r+self.image_size, c:c+self.image_size, :]
		assert train_data.shape == test_data.shape, f"[{index}]: do not have the same dimension"

		#make PyTorch happy, it accepts PIL image only
		train_img = Image.fromarray(train_data)
		test_img = Image.fromarray(test_data)

		#randomly horizontal flip the image with probability of 0.5
		random_val = np.random.uniform(0,1)
		if(random_val > 0.5):
			train_img = train_img.transpose(Image.FLIP_LEFT_RIGHT)
			test_img = test_img.transpose(Image.FLIP_LEFT_RIGHT)

		#randomly vertically flip the image with probability 0.5
		random_vertical = np.random.uniform(0,1)
		if random_vertical > 0.5:
			train_img = train_img.transpose(Image.FLIP_TOP_BOTTOM)
			test_img = test_img.transpose(Image.FLIP_TOP_BOTTOM)

		#transform each data to tensor with value range 0-1
		train_img = self.img_transforms(train_img)
		test_img = self.img_transforms(test_img)
		return test_img, train_img