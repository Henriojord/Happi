"""
Class representing a simple convolutional neural network-based classifier:
[[conv, ..., conv]->max pool, ..., [conv, ..., conv]->max pool]->[fc->...fc]->fc
"""

import torch

class SimpleClassifier(torch.nn.Module):
    def __init__(self, nb_filters, nb_layers, nb_blocks, dense, nb_class=185, input_size=256):
        """
        SimpleClassifier's constructor
        Args:
            nb_filters (int): Number of filters in the first convolutional layer (doubled after each max pooling operation)
            nb_layers (int): Number of convolutional layers before each max pooling operation
            nb_blocks (int): Number of max pooling operations
            dense (str): Fully-connected architecture (eg. '#neuron_layer1,#neuron_layer2,#neuron_layer3')
            nb_class (int): Number of classes (The Leafsnap-dataset includes 185 species)
            input_size (int): Image input size (considered to be a square. eg, 256x256)
        """

        super(SimpleClassifier, self).__init__()

        #Model's attributes
        self.nb_filters = nb_filters
        self.nb_layers = nb_layers
        self.nb_blocks = nb_blocks
        self.dense = [int(d) for d in dense.split(',')]
        self.nb_class = nb_class
        self.input_size = input_size

        #Encapsulate [conv, ..., conv]->max pool into a function
        def downsampling_block(in_dim, nb_f, nb_l):
            layers = []

            for n in range(nb_l):
                layers.append(torch.nn.Conv2d(in_dim, nb_f, (3, 3), padding=1))
                layers.append(torch.nn.ReLU())
                in_dim = nb_f

            layers.append(torch.nn.MaxPool2d((2, 2), (2, 2)))

            return layers

        layers = []
        prev_in = 3
        prev_f = self.nb_filters

        #Convolutional part (stack of downsampling_blocks)
        for n in range(self.nb_blocks):
            layers += downsampling_block(prev_in, prev_f, self.nb_layers)
            prev_in = prev_f
            prev_f *= 2

        self.conv = torch.nn.Sequential(*layers)

        #Fully-connected part (stack of fully-connected (Linear) layers)
        flatten = ((self.input_size//(2**self.nb_blocks))**2) * (prev_f // 2)

        layers = [torch.nn.Linear(flatten, self.dense[0]), torch.nn.ReLU()]

        if len(self.dense) > 1:
            for d in range(1, len(self.dense)):
                layers.append(torch.nn.Linear(self.dense[d - 1], self.dense[d]))
                layers.append(torch.nn.ReLU())
        self.fc = torch.nn.Sequential(*layers)

        #Classes' evidence layer
        self.evidence = torch.nn.Linear(self.dense[-1], self.nb_class)

        #Weight initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal(m.weight)

    def forward(self, x):
        #Convolutional part
        x = self.conv(x)

        #Flatten convolutional part's output
        x = x.view(x.size(0), -1)

        #Fully-connected part
        x = self.fc(x)

        #Evidences
        logits = self.evidence(x)

        return logits
