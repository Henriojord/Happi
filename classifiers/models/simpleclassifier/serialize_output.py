import argparse
import os
import torch
import pickle
import numpy as np
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

if not os.path.exists(os.path.join(args.directory, 'serialized_logits')):
    os.makedirs(os.path.join(args.directory, 'serialized_logits'))
if not os.path.exists(os.path.join(args.directory, 'serialized_logits', args.species)):
    os.makedirs(os.path.join(args.directory, 'serialized_logits', args.species))

cuda = torch.cuda.is_available()

testset = leafsnapdataset.LeafsnapDataset(args.root_dir, args.testset, (args.image_size, args.image_size))

model = simpleclassifier.SimpleClassifier(args.filter, args.layer, args.block, args.dense, testset.nb_class, args.image_size)
model.load_state_dict(torch.load(os.path.join(args.directory, 'serial', 'best_model')))
if cuda:
    model = model.cuda()
model.train(False)
print(model)

i = 0
dataloader = DataLoader(testset, batch_size=8, shuffle=True, num_workers=4)

for i_batch, sample in enumerate(dataloader):
    images = torch.tensor(sample['image']).float()
    if cuda:
        images = images.cuda()
    images = utils.processing.preprocess(images)
    logits = model(images)
    likelihood = torch.nn.functional.softmax(logits, 1)
    _, classe = torch.max(likelihood, 1)

    for c in range(len(classe)):
        #If model's answer is correct and the species is the one specified as argument
        if classe[c].item() == sample['label'][c].item() and sample['species'][c] == args.species:
            print(likelihood.shape)
            output = {k:(v, likelihood[c][v].item()) for k, v in testset.classes.items()}
            print(output)
            # with open(os.path.join(args.directory, 'serialized_logits', args.species, sample['name'][c]), 'wb') as f:
            #     pickle.dump(logits.cpu().detach().numpy(), f)
