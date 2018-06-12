import argparse
import pickle
import torch
import numpy as np

from dataset import leafsnapdataset

parser = argparse.ArgumentParser(description='')
parser.add_argument('--rd', dest='root_dir', type=str, default='/home/scom/data/umn64', help='Path to the images')
parser.add_argument('-s', dest='serial', type=str, default='', help='Path to a serialized output')
args = parser.parse_args()

dataset = leafsnapdataset.LeafsnapDataset(args.root_dir, 'data/summaries/testset', (256, 256))
with open('FILE', 'rb') as f:
    data = pickel.load(f)
data = torch.from_numpy(data)
likelihood = torch.nn.functional.softmax(data, 1)
_, classe = torch.max(likelihood, 1)
classe = classe.cpu().numpy()[0]
for s, c in list.items():
    if c == classe:
        species = s
print(species)
