import argparse
import os
import torch
import pickle
import numpy as np
from scipy import misc
from torch.utils.data import DataLoader

import utils.processing
from dataset import leafsnapdataset
import classifiers.models.simpleclassifier.model as simpleclassifier

parser = argparse.ArgumentParser(description='')

parser.add_argument('-m', dest='model', type=str, default='', help='Serialized model')
parser.add_argument('--tes', dest='testset', type=str, default='dataset/summaries/testset', help='Path to the testset summary')
parser.add_argument('--rd', dest='root_dir', type=str, default='/home/scom/leafsnap-dataset', help='Path to the images')
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

model = simpleclassifier.SimpleClassifier(args.filter, args.layer, args.block, args.dense, testset.nb_class, args.image_size)
model.load_state_dict(torch.load(os.path.join(args.directory, 'serial', 'best_model')))
if cuda:
    model = model.cuda()
model.train(False)
print(model)

i = 0
image = misc.imread('img/momiji.JPG')
image = misc.imresize(image, (256, 256))
image = image.reshape((1, 256, 256, 3))
image = torch.tensor(image).float()
if cuda:
    image = image.cuda()
image = utils.processing.preprocess(image)
logits = model(image)
likelihood = torch.nn.functional.softmax(logits, 1)
_, classe = torch.max(likelihood, 1)

output = {k:(v, likelihood[v].item()) for k, v in testset.classes.items()}

print(name)
best_1 = [0,0]
best_2 = [0,0]
best_3 = [0,0]
for k, v in output.items():
    if v[1] > best_1[1]:
        best_1[0] = k
        best_1[1] = v[1]
    elif v[1] > best_2[1]:
        best_2[0] = k
        best_2[1] = v[1]
    elif v[1] > best_3[1]:
        best_3[0] = k
        best_3[1] = v[1]
print(best_1)
print(best_2)
print(best_3)
print(output['Acer palmatum'])
            # with open(os.path.join(args.directory, 'serialized_logits', args.species, sample['name'][c]), 'wb') as f:
            #     pickle.dump(logits.cpu().detach().numpy(), f)
