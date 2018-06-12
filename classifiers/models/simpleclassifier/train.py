"""
SimpleClassifier's train script
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

import classifiers.models.simpleclassifier.model as simpleclassifier
import dataset.leafsnapdataset as leafsnapdataset
import utils.plot
import utils.processing

def train(model, datasets, optimizer, epoch, batch_size, directory):
    """
    Train the given model

    Args:
        model (classifiers.models.simpleclassifier.SimpleClassifier): Model to train
        datasets (dictionnary): Dictionnary containing the trainset and the testset ({'train': dataset.leafsnapdataset.LeafsnapDataset, 'test': dataset.leafsnapdataset.LeafsnapDataset})
        optimizer (torch.optim.Optimizer): Optimization algorithm
        epoch (int): Number of training epochs
        batch_size (int): Mini-batch size
        directory (str): Directory for storing train logs
    """

    cuda = torch.cuda.is_available()

    phase = ('train', 'test')
    best_accuracy = 0

    writer = SummaryWriter(os.path.join(directory, 'logs'))

    for e in range(epoch):
        for p in phase:
            running_loss = 0
            nb_correct = 0
            nb_sample = 0

            model.train(p == 'train')

            dataloader = DataLoader(datasets[p], batch_size=batch_size, shuffle=True, num_workers=4)

            for i_batch, sample in enumerate(tqdm(dataloader)):
                #Prepare batch
                images = torch.tensor(sample['image']).float()
                labels = sample['label'].long()

                if cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                images = utils.processing.preprocess(images) #Make pixels intensity range in [-1, 1]

                nb_sample += images.size(0)

                #Feed-forward pass
                logits = model(images)

                #Compute loss
                loss = torch.nn.functional.cross_entropy(logits, labels)
                if p == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()

                #Compute number of correct predictions
                likelihood = torch.nn.functional.softmax(logits, 1)
                _, classes = torch.max(likelihood, 1)
                nb_correct += torch.sum(classes == labels).item()

            epoch_loss = running_loss / nb_sample
            epoch_accuracy = (nb_correct / nb_sample) * 100

            writer.add_scalar('learning_curve/{}/loss'.format(p), epoch_loss, e)
            writer.add_scalar('learning_curve/{}/accuracy'.format(p), epoch_accuracy, e)

            print('Epoch {} ({}), Loss = {}, Accuracy = {}% ({} / {})'.format(e, p, epoch_loss, epoch_accuracy, nb_correct, nb_sample))

            if p == 'test' and epoch_accuracy > best_accuracy:
                torch.save(model.state_dict(), os.path.join(directory, 'serial', 'best_model'))
                best_accuracy = epoch_accuracy
                print('Model saved in {}'.format(os.path.join(directory, 'serial', 'best_model')))
                with open(os.path.join(directory, 'serial', 'save_history'), 'a') as f:
                    f.write('epoch: {}\tloss: {}\t accuracy: {}\n'.format(e, epoch_loss, epoch_accuracy))

    writer.export_scalars_to_json(os.path.join(directory, 'logs', 'scalars.json'))
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    #Training arguments
    parser.add_argument('--trs', dest='trainset', type=str, default='dataset/summaries/trainset', help='Path to the trainset summary')
    parser.add_argument('--tes', dest='testset', type=str, default='dataset/summaries/testset', help='Path to the testset summary')
    parser.add_argument('--rd', dest='root_dir', type=str, default='home/jordan/leafsnap', help='Path to the images')
    parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--ep', dest='epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--dir', dest='directory', type=str, default='leafsnap_simpleclassifier', help='Directory to store results')
    parser.add_argument('--ims', dest='image_size', type=int, default=256, help='Image size')

    #Model arguments
    parser.add_argument('-f', dest='filter', type=int, default=32, help='Number of filters')
    parser.add_argument('-b', dest='block', type=int, default=4, help='Number of downsampling blocks')
    parser.add_argument('-l', dest='layer', type=int, default=2, help='Number of layers per block')
    parser.add_argument('-d', dest='dense', type=str, default='128,128', help='Fully-connected architecture')

    args = parser.parse_args()

    #Create directories if it don't exists
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    if not os.path.exists(os.path.join(args.directory, 'serial')):
        os.makedirs(os.path.join(args.directory, 'serial'))
    if not os.path.exists(os.path.join(args.directory, 'logs')):
        os.makedirs(os.path.join(args.directory, 'logs'))

    #Datasets
    trainset = leafsnapdataset.LeafsnapDataset(args.root_dir, args.trainset, (args.image_size, args.image_size))
    testset = leafsnapdataset.LeafsnapDataset(args.root_dir, args.testset, (args.image_size, args.image_size))
    datasets = {'train': trainset, 'test': testset}

    model = simpleclassifier.SimpleClassifier(args.filter, args.layer, args.block, args.dense, trainset.nb_class, args.image_size)
    if torch.cuda.is_available():
        model = model.cuda()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    train(model, datasets, optimizer, args.epoch, args.batch_size, args.directory)
