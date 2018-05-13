"""
SimpleClassifier's train script
"""

import torch
from torch.utils.data import DataLoader

import classifiers.models.simpleclassifier as simpleclassifier
import dataset.leafsnapdataset as leafsnapdataset
import utils.plot
import utils.processing

def train(model, datasets, optimizer, epoch, directory):
    """
    Train the given model

    Args:
        model (classifiers.models.simpleclassifier.SimpleClassifier): Model to train
        datasets (dictionnary): Dictionnary containing the trainset and the testset ({'train': dataset.leafsnapdataset.LeafsnapDataset, 'test': dataset.leafsnapdataset.LeafsnapDataset})
        optimizer (torch.optim.Optimizer): Optimization algorithm
        epoch (int): Number of training epochs
        directory (str): Directory for storing train logs
    """

    phase = ('train', 'test')

    hist_loss = {p:[] for p in phase}
    hist_accuracy = {p:[] for p in phase}

    for e in range(epoch):
        for p in phase:
            running_loss = 0
            nb_correct = 0
            nb_sample = 0

            dataloader = DataLoader(datasets[p], batch_size=batch_size, shuffle=True, num_workers=4)

            for i_batch, sample in enumerate(tqdm(dataloader)):
                #Prepare batch
                images = torch.tensor(sample['image']).float().cuda()
                images = utils.preprocess(images)

                labels = sample['label'].long().cuda()

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
                _, classes = torch.max(logits, 1)
                nb_correct += torch.sum(classes == labels).item()

            hist_loss[p].append(running_loss / nb_sample)
            hist_accuracy[p].append((nb_correct / nb_sample) * 100)

            print('Epoch {} ({}), Loss = {}, Accuracy = {}% ({} / {})'.format(e, p, hist_loss[p][-1], hist_accuracy[p][-1], nb_correct, nb_sample))

            if e % 5 == 0 and p == 'test':
                utils.plot.plot_learning_curves(hist_loss, hist_accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    #Training arguments
    parser.add_argument('--trs', dest='trainset', type=str, default='dataset/summaries/leafsnaptrainset', help='Path to the trainset summary')
    parser.add_argument('--tes', dest='testset', type=str, default='dataset/summaries/leafsnaptestset', help='Path to the testset summary')
    parser.add_argument('--rd', dest='root_dir', type=str, default='home/jordan/leafsnap', help='Path to the images')
    parser.add_argument('--bs', dest='batch_size', type=int, default=16, help='Mini batch size')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--ep', dest='epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--dir', dest='directory', type=str, default='leafsnap_simpleclassifier', help='Directory to store results')
    parser.add_argument('--ims', dest='image_size', type=str, default='256,256', help='Image size')

    #Model arguments
    parser.add_argument('-f', dest='filter', type=int, default=32, help='Number of filters')
    parser.add_argument('-b', dest='block', type=int, default=4, help='Number of downsampling blocks')
    parser.add_argument('-l', dest='layer', type=int, default=2, help='Number of layers per block')
    parser.add_argument('-d', dest='dense', type=int, default='128,128', help='Fully-connected architecture')

    args = parser.parse_args()

    #Datasets
    trainset = leafsnapdataset.LeafsnapDataset(args.trainset, (args.image_size, args.image_size))
    testset = leafsnapdataset.LeafsnapDataset(args.testset, (args.image_size, args.image_size))
    datasets = {'train': trainset, 'test': testset}

    model = SimpleClassifier(args.filter, args.layer, args.block, args.dense, trainset.nb_class, args.image_size)
    model = model.cuda()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    train(model, datasets, optimizer, args.directory)
