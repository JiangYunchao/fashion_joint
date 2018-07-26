import os
from PIL import Image
import torch
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import config as cfg
import pickle
import sys
import numpy as np
import os.path

sys.path.insert(0, '/share1/home/jiang/polyvore-kit')
from utils.data_utils import load_pkl
from utils.data_utils import DataFile

sys.path.insert(0, '/share1/home/jiang/test_180120/')


def default_loader(img_path):
    return Image.open(img_path).convert("RGB")

class Basic(Dataset):

    def __init__(self,
                 img_dir='/share2/home/zhi/data/polyvore/processed/images/291x291/',
                 list_dir='/share1/home/jiang/data/polyvore/processed/image_list/',
                 pkl_dir='/share1/home/jiang/data/processed/pickles',
                 tuples='/share1/home/jiang/data/processed/tuples',
                 text_dir='/share1/home/jiang/pytorch/text_file/',
                 text_list_dir='/share1/home/jiang/pytorch/text_file/',
                 loader = default_loader,
                 phase=None):
        self.img_dir = img_dir
        self.list_dir = list_dir
        self.pkl_dir = pkl_dir
        self.tuples = tuples
        self.text_dir = text_dir
        self.text_list_dir = text_list_dir
        self.phase = phase
        self.loader = loader
        
    def getlist(self, text_list_dir):
        pos_text = []
        pos_list = []
        neg_list = []
        uidx_list = []
        if (self.phase == "train") :
            with open (text_list_dir, 'r') as f:
                for line in f.readlines():
                    line.rstrip()
                    pos = line.split()[0]
                    pos_text.append(pos)
                    pos_list.append(pos_text)
                    pos_text = []
                    idx = int(pos.split('_')[1])
                    uidx_list.append(idx)
                    neg_list.append(line.split()[1:3])
        else:
            with open (text_list_dir, 'r') as f:
                for line in f.readlines():
                    line.rstrip()
                    pos = line.split()[0]
                    pos_text.append(pos)
                    pos_list.append(pos_text)
                    pos_text = []
                    idx = int(pos.split('_')[1])
                    uidx_list.append(idx)
                    neg_list.append(line.split()[1:])

        return uidx_list, pos_list, neg_list

    def getlist_img(self, pkl_dir, tuples, image_list):
        #fashion_sets, fashion_items = load_pkl(pkl_dir)
        datafile = DataFile(tuples, image_list)
        image_list = datafile.image_list
        positive_tuples, negative_tuples = datafile.get_tuples(self.phase, repeated=False)
        return positive_tuples, negative_tuples, image_list


    def getdata(self, filename):
        rt = os.path.join(self.text_dir, self.phase, filename)
        with open(rt, 'r') as f:
            n = 0
            oft_text = []
            for line in f.readlines():
                #line = line.rstrip("\n").rstrip("\t")
                try:
                    item = line.split('||||')[2]  #.split('\t')
                except:
                    print(len(f.readlines()), rt, n)
                n += 1
                cate = line.split('||||')[1]
                oft_text.append(item)
        return oft_text

    def getdata_img(self, oft):
        img_oft = []
        for item in oft:
            img = self.loader(os.path.join(self.img_dir, item))
            img1 = self.transform1(img)
            img_oft.append(img1)
        return img_oft


class PolyvoreDataset(Basic):
    """Class for polyvore data set."""

    def __init__(self,
                 img_dir='/share2/home/zhi/data/polyvore/processed/images/291x291/',
                 list_dir='/share2/home/zhi/data/polyvore/processed/variable/image_list/',
                 pkl_dir='/share1/home/jiang/data/processed/pickles',
                 tuples='/share1/home/jiang/data/processed/tuples',
                 text_dir='/share1/home/jiang/pytorch/text_file/',
                 text_list_dir='/share1/home/jiang/pytorch/text_file/',
                 loader=None,
                 phase = None):
        """Initialize a PolyvoreDataset.
        Parameters
        ----------
        image_dir: where to load image for each category.
        tuple_dir: to load positive outfits
        list_dir: to load image list
        phase: phase for train val or test
        transforms: transformation for data
        Algorithms
        ----------
        1. hard mode: generate negative examples from pisitive tuples that
                      created by other user.
        2. fixed mode: use negatived exmapled saved in file.
        3. match mode: generated negative exampled from random with items from
                       the positive set of this user. This is for learning the
                       matching term in score.
        """
        super(PolyvoreDataset, self).__init__(img_dir, list_dir,pkl_dir, tuples,
                                              text_dir, text_list_dir,loader, phase)


        path = os.path.join(self.text_list_dir, self.phase + '_combine.txt')
        self.uidx_list, self.pos_txt, self.neg_txt = self.getlist(path)
        self.num_oft = len(self.pos_txt)
        self.num_users = len(set(self.uidx_list))
        fr = open("/share1//home/jiang/fashion_net/sen2vec/text_vec_" + phase + "_padding_norm.pkl", "rb")
        self.d = pickle.load(fr)
        self.padding = np.array(self.d["null"])
        self.transform1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # self.n_pos = 0 #count number of items without text_vec
        # self.n_neg = 0 #count number of items without text_vec
    def get_tuples(self, index):
        """Get one pair outfits.
        Return
        -----
        uidx: the index of user
        posioft: the positive outfit
        negaoft: the negative outfit
        """


        uidx = self.uidx_list[index]
        posilist = self.pos_txt[index]
        negalist = self.neg_txt[index]


        return uidx, posilist, negalist
    def conv2vec(self, oft_list):
        text_list = []
        img_list = []
        for oft in oft_list:
            p = self.getdata(oft)
            oft_text = []
            oft_img = self.getdata_img(p)
            for item in p:
                if item in self.d:
                    temp = np.array(self.d[item])
                    oft_text.append(np.squeeze(temp))
                else:
                    oft_text.append(np.squeeze(self.padding))
                    #self.n_pos += 1
                    #print("pos: ", item)
            text_list.append(oft_text)
            img_list.append(oft_img)
        return text_list, img_list

    def __getitem__(self, index):
        """Get one tuple of examples by index."""
        uidx, pos, neg = self.get_tuples(index)
        posilist_text, posilist_img = self.conv2vec(pos)
        negalist_text, negalist_img = self.conv2vec(neg)


        #print("posi_ft_len:{}".format(posi_ft))
        # if len(posi_ft) != 3:
        #     print("lenerror!pos:", pos_rt, posi_ft)
        # if len(nega_ft) != 3:
        #     print("lenerror!neg:", neg_rt, nega_ft)
        # if np.shape(posi_ft[0]) != np.shape(posi_ft[1]):
        #     print("hiddenerror!root:{},posi:{},ft0:{},ft1{}".format(pos_rt, posi_ft, np.shape(posi_ft[0]), np.shape(posi_ft[1])))
        # if np.shape(posi_ft[0]) != np.shape(posi_ft[2]):
        #     print("hiddenerror!root:{},posi:{},ft0:{},ft2{}".format(pos_rt, posi_ft, np.shape(posi_ft[0]), np.shape(posi_ft[2])))
        # if np.shape(nega_ft[0]) != np.shape(nega_ft[1]):
        #     print("hiddenerror!root:{},nega:{},ft0:{},ft1{}".format(neg_rt, nega_ft, np.shape(posi_ft[0]), np.shape(posi_ft[1])))
        # if np.shape(nega_ft[0]) != np.shape(nega_ft[2]):
        #     print("hiddenerror!root:{},nega:{},ft0:{},ft2{}".format(neg_rt, nega_ft, np.shape(posi_ft[0]), np.shape(posi_ft[2])))
        return posilist_text, negalist_text, posilist_img, negalist_img, uidx

    def __len__(self):
        """Return the size of dataset."""
        return self.num_oft

#def collate_fn(batch):
 #   batch.sort(key=lambda x: len(x[2]), reverse=True)
  #  posi_ft, nega_ft, uidx = zip(*batch)
   # pad_label = []
    #lens = []
    #max_len = len(uidx[0])
    #for i in range(len(uidx)):
     #   temp_label = [0] * max_len
      #  temp_label[:len(uidx[i])] = uidx[i]
       # pad_label.append(temp_label)
        #lens.append(len(uidx[i]))
    #return posi_ft, nega_ft, uidx

class PolyvoreDataLoader(object):
    """Date loader for Ployvore."""

    def __init__(self,
                 args,
                 phase=None,
                 variable=False,
                 triplet=False,
                 evaluate=False):
        """Initialize a data loder."""


        dataset = PolyvoreDataset(
            args.img_dir,
            args.list_dir,
            args.pkl_dir,
            args.tuple_dir,
            args.text_dir,
            args.text_list_dir,
            loader=default_loader,
            phase=phase
            )
        self.loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.works,
            shuffle=True,
            pin_memory=True,
            #collate_fn=collate_fn
        )
        self.num_users = dataset.num_users
        
