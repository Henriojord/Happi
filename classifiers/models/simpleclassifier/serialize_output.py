import argparse
import os
import torch
import numpy as np

import utils
from dataset import leafsnapdataset
from classifiers.models import simpleclassifier

parser = argparse.ArgumentParser(description='')

parser.add_argument('-m', dest='model', type=str, default='', help='Serialized model')
parser.add_argument('--tes', dest='testset', type=str, default='data/summaries/testset', help='Path to the testset summary')
parser.add_argument('--rd', dest='root_dir', type=str, default='/home/scom/data/umn64', help='Path to the images')
parser.add_argument('--dir', dest='directory', type=str, default='/home/scom/Documents/happi_exp1', help='Directory to store results')
parser.add_argument('--ims', dest='image_size', type=int, default=256, help='Image size')
parser.add_argument('-s', dest='species', type=str, default='', help='Species to serialize')
#Model arguments
parser.add_argument('-f', dest='filter', type=int, default=16, help='Number of filters')
parser.add_argument('-b', dest='block', type=int, default=6, help='Number of downsampling blocks')
parser.add_argument('-l', dest='layer', type=int, default=2, help='Number of layers per block')
parser.add_argument('-d', dest='dense', type=str, default='128,128,128', help='Fully-connected architecture')
args = parser.parse_args()

cuda = torch.cuda.is_available()

testset = leafsnapdataset.LeafsnapDataset(args.root_dir, args.testset, (args.image_size, args.image_size))

model = simpleclassifier.SimpleClassifier(args.filter, args.layer, args.block, args.dense, trainset.nb_class, args.image_size)
if cuda:
    model = model.cuda()

print(model)

i = 0
while '' not in testset[i]['name']:
    i += 1

image = torch.tensor(testset[i]['image']).float()
if cuda:
    image = image.cuda()
image = utils.processing.preprocess(image)

logits = model(image)
likelihood = torch.nn.functional.softmax(logits, 1)
_, classe = torch.max(likelihood, 1)

name = testset[i]['name'].split('/')[-1][:-4]
with open(name, 'w') as f:
    f.write(str(classe))
