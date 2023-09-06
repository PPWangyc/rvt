import json
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset

class FaceF_whole_video_2_labels():
    def __init__(self, img_path, config_path='./configs/data.json', train=True, transform = None, mask_signal=True, select=False, aug=False) -> None:
        self.img_path = img_path
        self.config_path = config_path
        self.transform_img = transform
        self.transform_label = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.train=train
        self.select = select
        self.aug = aug
        self.mask_signal = mask_signal
        self.preprocess()
        

    def preprocess(self):
        # read train_set from data.json
        configs = json.load(open(self.config_path))
        if self.train:
            train_set = configs['train_set']
        else:
            train_set = configs['test_set']
        # get train_set path join with img_path
        if self.select:
            train_set = configs['select_test_set']
        train_set = [os.path.join(self.img_path, file) for file in train_set]
        # get all folders name in train_set
        train_set= [os.path.join(file, folder) for file in train_set for folder in os.listdir(file)]
        # exclude .mp4 files
        train_set = [file for file in train_set if not file.endswith('.mp4')]

        self.train_path = []
        self.label_path = []
        self.mask_path = []
        
        self.train_path_dict = {}
        self.label_path_dict = {}
        self.mask_path_dict = {}

        dict_key = 0

        for file_path in train_set:
            # get txt file under file_path
            # self.label_path.append([os.path.join(file_path, txt) for txt in os.listdir(file_path) if txt.endswith('value.txt')][0])
            label_txt_path = [os.path.join(file_path, txt) for txt in os.listdir(file_path) if txt.endswith('value.txt')][0]
            mask_txt_path = [os.path.join(file_path, txt) for txt in os.listdir(file_path) if txt.endswith('label.txt')][0]
            # self.mask_path.append(mask_txt_path)
            # # get jpg file under file_path
            # map[mask_txt_path]=[os.path.join(file_path, jpg) for jpg in os.listdir(file_path) if jpg.endswith('.jpg')]
            total_time,fatigue = self.read_label(label_txt_path)

            for jpg in os.listdir(file_path):
                if jpg.endswith('.jpg'):
                    self.train_path.append(os.path.join(file_path, jpg))
                    self.label_path.append(label_txt_path)
                    self.mask_path.append(mask_txt_path)
            # sort train_path according to time, where time is an integer. sort label and mask path according to train_path index
            self.train_path.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
            self.train_path_dict[dict_key] = self.train_path
            self.label_path_dict[dict_key] = self.label_path
            self.mask_path_dict[dict_key] = self.mask_path
            
            self.train_path = []
            self.label_path = []
            self.mask_path = []
            dict_key += 1
        # if self.train_path_dict contains empty list, delete it, and make sure the key is continuous
        for key in list(self.train_path_dict.keys()):
            if len(self.train_path_dict[key]) == 0:
                del self.train_path_dict[key]
                del self.label_path_dict[key]
                del self.mask_path_dict[key]
        
        # change dict key to continuous
        self.train_path_dict = {i:self.train_path_dict[key] for i, key in enumerate(self.train_path_dict.keys())}
        self.label_path_dict = {i:self.label_path_dict[key] for i, key in enumerate(self.label_path_dict.keys())}
        self.mask_path_dict = {i:self.mask_path_dict[key] for i, key in enumerate(self.mask_path_dict.keys())}

        assert(len(self.train_path_dict) == len(self.label_path_dict)== len(self.mask_path_dict)) 
        self.num_videos = len(self.train_path_dict)

    
    def read_label(self, label_path):
        # read first line of label_path
        with open(label_path, 'r') as f:
            line = f.readline()
            # get the first 5 numbers
            fatigue = line.split('fatigue:')[1][1:-2]
            temp2 = line.split(' ')[1]
        
        # strip all ' in string fatigue
        fatigue = fatigue.replace("'", "").split(',')
        # delete all "'" and ' ' in fatigue tuple
        for i in range(len(fatigue)):
            fatigue[i] = fatigue[i].replace("'", "").replace(' ', '')
        # convert string to numpy array, if string is empty convert to np.nan
        fatigue = np.array([np.nan if i == '' else float(i) for i in fatigue])
        total_time = int(float(temp2.split(':')[-1].strip(',')))
        return total_time, fatigue

    def read_mask(self, mask_path, train_path):
        
        # get train file time
        time = train_path.split('/')[-1].split('.')[0]
        # read mask file, and find the line start equal to time
        with open(mask_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.split(' ')[0] == time:
                return int(line.split(' ')[1][0])

    def __getitem__(self, index):
        img_paths, label_path, mask_path = self.train_path_dict[index], self.label_path_dict[index][0], self.mask_path_dict[index][0]
        
        _, fatigue = self.read_label(label_path)
        label = [float(fatigue[0]), float(fatigue[13])]
        label = [[0 if l < 2 else 1 for l in label]]
        label = torch.tensor(label)

        videos = []
        frames = []
        cur = 1
        for img_path in img_paths:
            time = int(img_path.split('/')[-1].split('.')[0])
            if time > cur * 300:
                videos.append(torch.stack(frames, dim = 0))
                frames = []
                cur += 1
            image = Image.open(img_path)
            mask = self.read_mask(mask_path, img_path)
            if mask is None:
                mask = 1
            # add maks signal to image
            image = torch.cat((self.transform_img(image), torch.ones(3, 112, 1) * mask), dim=2)
            frames.append(image)
        if len(videos) == 0:
            videos.append(torch.stack(frames, dim = 0))
        # remove the video in videos if its length is less than 16
        i = 0
        while True:
            if len(videos[i]) < 16:
                del videos[i]
                i -= 1
            else:
                videos[i] = videos[i][:16]
            i += 1
            if i == len(videos):
                break

        if len(videos) == 0:
            return [], label.T
        return torch.stack(videos, dim = 0), label.T
    
    def __len__(self):
        """Return the number of images."""
        return self.num_videos

class ToDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.labels_mean = labels.float().mean()
        self.labels_std = labels.float().std()
        self.labels_median = labels.float().median()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    

    def __init__(self, img_path, config_path='./configs/data.json', train=True, transform = None, mask_signal=True, select=False, aug=False) -> None:
        self.img_path = img_path
        self.config_path = config_path
        self.transform_img = transform
        self.transform_label = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.train=train
        self.select = select
        self.aug = aug
        self.mask_signal = mask_signal
        self.preprocess()
        

    def preprocess(self):
        # read train_set from data.json
        configs = json.load(open(self.config_path))
        if self.train:
            train_set = configs['train_set']
        else:
            train_set = configs['test_set']
        # get train_set path join with img_path
        if self.select:
            train_set = configs['select_test_set']
        train_set = [os.path.join(self.img_path, file) for file in train_set]
        # get all folders name in train_set
        train_set= [os.path.join(file, folder) for file in train_set for folder in os.listdir(file)]
        # exclude .mp4 files
        train_set = [file for file in train_set if not file.endswith('.mp4')]

        self.train_path = []
        self.label_path = []
        self.mask_path = []
        
        self.train_path_dict = {}
        self.label_path_dict = {}
        self.mask_path_dict = {}

        dict_key = 0

        for file_path in train_set:
            # get txt file under file_path
            # self.label_path.append([os.path.join(file_path, txt) for txt in os.listdir(file_path) if txt.endswith('value.txt')][0])
            label_txt_path = [os.path.join(file_path, txt) for txt in os.listdir(file_path) if txt.endswith('value.txt')][0]
            mask_txt_path = [os.path.join(file_path, txt) for txt in os.listdir(file_path) if txt.endswith('label.txt')][0]
            # self.mask_path.append(mask_txt_path)
            # # get jpg file under file_path
            # map[mask_txt_path]=[os.path.join(file_path, jpg) for jpg in os.listdir(file_path) if jpg.endswith('.jpg')]
            total_time,fatigue = self.read_label(label_txt_path)

            for jpg in os.listdir(file_path):
                if jpg.endswith('.jpg'):
                    self.train_path.append(os.path.join(file_path, jpg))
                    self.label_path.append(label_txt_path)
                    self.mask_path.append(mask_txt_path)
            # sort train_path according to time, where time is an integer. sort label and mask path according to train_path index
            self.train_path.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
            self.train_path_dict[dict_key] = self.train_path
            self.label_path_dict[dict_key] = self.label_path
            self.mask_path_dict[dict_key] = self.mask_path
            
            self.train_path = []
            self.label_path = []
            self.mask_path = []
            dict_key += 1
        # if self.train_path_dict contains empty list, delete it, and make sure the key is continuous
        for key in list(self.train_path_dict.keys()):
            if len(self.train_path_dict[key]) == 0:
                del self.train_path_dict[key]
                del self.label_path_dict[key]
                del self.mask_path_dict[key]
        
        # change dict key to continuous
        self.train_path_dict = {i:self.train_path_dict[key] for i, key in enumerate(self.train_path_dict.keys())}
        self.label_path_dict = {i:self.label_path_dict[key] for i, key in enumerate(self.label_path_dict.keys())}
        self.mask_path_dict = {i:self.mask_path_dict[key] for i, key in enumerate(self.mask_path_dict.keys())}

        assert(len(self.train_path_dict) == len(self.label_path_dict)== len(self.mask_path_dict)) 
        self.num_videos = len(self.train_path_dict)

    
    def read_label(self, label_path):
        # read first line of label_path
        with open(label_path, 'r') as f:
            line = f.readline()
            # get the first 5 numbers
            fatigue = line.split('fatigue:')[1][1:-2]
            temp2 = line.split(' ')[1]
        
        # strip all ' in string fatigue
        fatigue = fatigue.replace("'", "").split(',')
        # delete all "'" and ' ' in fatigue tuple
        for i in range(len(fatigue)):
            fatigue[i] = fatigue[i].replace("'", "").replace(' ', '')
        # convert string to numpy array, if string is empty convert to np.nan
        fatigue = np.array([np.nan if i == '' else float(i) for i in fatigue])
        total_time = int(float(temp2.split(':')[-1].strip(',')))
        return total_time, fatigue

    def read_mask(self, mask_path, train_path):
        
        # get train file time
        time = train_path.split('/')[-1].split('.')[0]
        # read mask file, and find the line start equal to time
        with open(mask_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.split(' ')[0] == time:
                return int(line.split(' ')[1][0])

    def __getitem__(self, index):
        img_paths, label_path, mask_path = self.train_path_dict[index], self.label_path_dict[index][0], self.mask_path_dict[index][0]
        
        _, fatigue = self.read_label(label_path)
        label = [float(fatigue[0]), float(fatigue[13])]
        # label = [[0 if l < 2 else 1 for l in label]]
        for l in range(len(label)):
            if label[l] < 2:
                label[l] = [0]
            elif label[l] < 5:
                label[l] = [1]
            else:
                label[l] = [2]
        label = torch.tensor(label).T
        videos = []
        frames = []
        cur = 1
        for img_path in img_paths:
            time = int(img_path.split('/')[-1].split('.')[0])
            if time > cur * 300:
                videos.append(torch.stack(frames, dim = 0))
                frames = []
                cur += 1
            image = Image.open(img_path)
            mask = self.read_mask(mask_path, img_path)
            # add maks signal to image
            image = torch.cat((self.transform_img(image), torch.ones(3, 112, 1) * mask), dim=2)
            frames.append(image)
        if len(videos) == 0:
            videos.append(torch.stack(frames, dim = 0))
        # remove the video in videos if its length is less than 16
        i = 0
        while True:
            if len(videos[i]) < 16:
                del videos[i]
                i -= 1
            else:
                videos[i] = videos[i][:16]
            i += 1
            if i == len(videos):
                break

        if len(videos) == 0:
            return [], label.T
        return torch.stack(videos, dim = 0), label.T
    
    def __len__(self):
        """Return the number of images."""
        return self.num_videos


    def __init__(self, img_path, config_path='./configs/data.json', train=True, transform = None, mask_signal=True, select=False, aug=False) -> None:
        self.img_path = img_path
        self.config_path = config_path
        self.transform_img = transform
        self.transform_label = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.train=train
        self.select = select
        self.aug = aug
        self.mask_signal = mask_signal
        self.preprocess()
        

    def preprocess(self):
        # read train_set from data.json
        configs = json.load(open(self.config_path))
        if self.train:
            train_set = configs['train_set']
        else:
            train_set = configs['test_set']
        # get train_set path join with img_path
        if self.select:
            train_set = configs['select_test_set']
        train_set = [os.path.join(self.img_path, file) for file in train_set]
        # get all folders name in train_set
        train_set= [os.path.join(file, folder) for file in train_set for folder in os.listdir(file)]
        # exclude .mp4 files
        train_set = [file for file in train_set if not file.endswith('.mp4')]

        self.train_path = []
        self.label_path = []
        self.mask_path = []
        
        self.train_path_dict = {}
        self.label_path_dict = {}
        self.mask_path_dict = {}

        dict_key = 0

        for file_path in train_set:
            # get txt file under file_path
            # self.label_path.append([os.path.join(file_path, txt) for txt in os.listdir(file_path) if txt.endswith('value.txt')][0])
            label_txt_path = [os.path.join(file_path, txt) for txt in os.listdir(file_path) if txt.endswith('value.txt')][0]
            mask_txt_path = [os.path.join(file_path, txt) for txt in os.listdir(file_path) if txt.endswith('label.txt')][0]
            # self.mask_path.append(mask_txt_path)
            # # get jpg file under file_path
            # map[mask_txt_path]=[os.path.join(file_path, jpg) for jpg in os.listdir(file_path) if jpg.endswith('.jpg')]
            total_time,fatigue = self.read_label(label_txt_path)

            for jpg in os.listdir(file_path):
                if jpg.endswith('.jpg'):
                    self.train_path.append(os.path.join(file_path, jpg))
                    self.label_path.append(label_txt_path)
                    self.mask_path.append(mask_txt_path)
            # sort train_path according to time, where time is an integer. sort label and mask path according to train_path index
            self.train_path.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
            self.train_path_dict[dict_key] = self.train_path
            self.label_path_dict[dict_key] = self.label_path
            self.mask_path_dict[dict_key] = self.mask_path
            
            self.train_path = []
            self.label_path = []
            self.mask_path = []
            dict_key += 1
        # if self.train_path_dict contains empty list, delete it, and make sure the key is continuous
        for key in list(self.train_path_dict.keys()):
            if len(self.train_path_dict[key]) == 0:
                del self.train_path_dict[key]
                del self.label_path_dict[key]
                del self.mask_path_dict[key]
        
        # change dict key to continuous
        self.train_path_dict = {i:self.train_path_dict[key] for i, key in enumerate(self.train_path_dict.keys())}
        self.label_path_dict = {i:self.label_path_dict[key] for i, key in enumerate(self.label_path_dict.keys())}
        self.mask_path_dict = {i:self.mask_path_dict[key] for i, key in enumerate(self.mask_path_dict.keys())}

        assert(len(self.train_path_dict) == len(self.label_path_dict)== len(self.mask_path_dict)) 
        self.num_videos = len(self.train_path_dict)

    
    def read_label(self, label_path):
        # read first line of label_path
        with open(label_path, 'r') as f:
            line = f.readline()
            # get the first 5 numbers
            fatigue = line.split('fatigue:')[1][1:-2]
            temp2 = line.split(' ')[1]
        
        # strip all ' in string fatigue
        fatigue = fatigue.replace("'", "").split(',')
        # delete all "'" and ' ' in fatigue tuple
        for i in range(len(fatigue)):
            fatigue[i] = fatigue[i].replace("'", "").replace(' ', '')
        # convert string to numpy array, if string is empty convert to np.nan
        fatigue = np.array([np.nan if i == '' else float(i) for i in fatigue])
        total_time = int(float(temp2.split(':')[-1].strip(',')))
        return total_time, fatigue

    def read_mask(self, mask_path, train_path):
        
        # get train file time
        time = train_path.split('/')[-1].split('.')[0]
        # read mask file, and find the line start equal to time
        with open(mask_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.split(' ')[0] == time:
                return int(line.split(' ')[1][0])

    def __getitem__(self, index):
        img_paths, label_path, mask_path = self.train_path_dict[index], self.label_path_dict[index][0], self.mask_path_dict[index][0]
        _, fatigue = self.read_label(label_path)
        label = [float(fatigue[0]), float(fatigue[13])]
        label = [[0 if l < 2 else 1 for l in label]]
        label = torch.tensor(label)

        videos = []
        frames = []
        cur = 1
        for img_path in img_paths:
            time = int(img_path.split('/')[-1].split('.')[0])
            if time > cur * 300:
                videos.append(torch.stack(frames, dim = 0))
                frames = []
                cur += 1
            image = Image.open(img_path)
            mask = self.read_mask(mask_path, img_path)
            # add maks signal to image
            if mask is None:
                mask = 1
            image = torch.cat((self.transform_img(image), torch.ones(3, 112, 1) * mask), dim=2)
            frames.append(image)
        if len(videos) == 0:
            videos.append(torch.stack(frames, dim = 0))
        # remove the video in videos if its length is less than 16
        i = 0
        while True:
            if len(videos[i]) < 16:
                del videos[i]
                i -= 1
            else:
                videos[i] = videos[i][:16]
            i += 1
            if i == len(videos):
                break

        if len(videos) == 0:
            return [], label.T, label_path
        return torch.stack(videos, dim = 0), label.T, label_path
    
    def __len__(self):
        """Return the number of images."""
        return self.num_videos