"""
Define the LeafsnapDataset class that handle data from the Leafsnap-dataset (http://leafsnap.com/dataset/)
"""

import os
from scipy import misc
from torch.utils.data import Dataset

class LeafsnapDataset(Dataset):
    def __init__(self, summary, resize=(256, 256)):
        """
        LeavesDataset constructor
        Args:
            summary (string): Path the dataset summary file
            resize (tuple of int): Shape for image resize
        """

        self.resize = resize

        with open(summary, 'r') as f:
            data = f.read().split('\n')[:-1]

        #Images
        self.data = [d.split('\t') for d in data]

        #Labels
        species = sorted(set([d[1] for d in self.data]))
        one_hots = [[0] * len(species) for i in range(len(species))]
        for i in range(len(species)):
            one_hots[i][i] = 1
        self.classes = {species[i]:i for i in range(len(species))}
        self.one_hots = {species[i]:one_hots[i] for i in range(len(species))}
        self.nb_class = len(self.classes.keys())

    def __len__(self):
        """
        Return the dataset's length
        """

        return len(self.data)

    def __getitem__(self, idx):
        """
        Return the specified sample
        """

        #image
        image = misc.imread(os.path.join('leafsnap-dataset', self.data[idx][0]))
        image = misc.imresize(image, self.resize)

        #label
        species = self.data[idx][1]
        one_hot = self.one_hots[species]
        label = self.classes[species]

        #sample
        sample = {'image': image, 'label': label, 'one_hot':one_hot, 'species': species}

        return sample
