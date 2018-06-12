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
# for t in testset:
    images = torch.tensor(sample['image']).float()
    if cuda:
        images = images.cuda()
    images = utils.processing.preprocess(images)
    logits = model(images)
    likelihood = torch.nn.functional.softmax(logits, 1)
    _, classe = torch.max(likelihood, 1)
    #classe = classe.item()

    for c in range(len(classe)):
        if classe[c].item() == sample['label'][c].item() and sample['species'][c] == args.species:
            with open(sample['name'][c], 'wb') as f:
                pickle.dump(logits.cpu().detach().numpy(), f)

    # if args.species in t['species']:
    #
    #     logits = model(image)
    #     likelihood = torch.nn.functional.softmax(logits, 1)
    #     _, classe = torch.max(likelihood, 1)
    #     classe = classe.item()
    #
    #     print(classe, testset.classes[testset.data[i][1]], testset.data[i][1])
    #     name = testset.data[i][0].split('/')[-1][:-4]
    #     if classe == testset.classes[testset.data[i][1]]:
    #         name = 'error_' + name
    #     with open(name, 'wb') as f:
    #         pickle.dump(logits.cpu().detach().numpy(), f)


    # i += 1
    # if args.species in t['species']:
    #     image = torch.tensor(t['image']).float()
    #     if cuda:
    #         image = image.cuda()
    #     image = utils.processing.preprocess(image.view(1, 256, 256, 3))
    #
    #     logits = model(image)
    #     likelihood = torch.nn.functional.softmax(logits, 1)
    #     _, classe = torch.max(likelihood, 1)
    #     classe = classe.item()
    #
    #     print(classe, testset.classes[testset.data[i][1]], testset.data[i][1])
    #     name = testset.data[i][0].split('/')[-1][:-4]
    #     if classe == testset.classes[testset.data[i][1]]:
    #         name = 'error_' + name
    #     with open(name, 'wb') as f:
    #         pickle.dump(logits.cpu().detach().numpy(), f)
    # i += 1
