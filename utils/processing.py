"""
Define a set of data processing functions
"""

def preprocess(x):
    """
    Preprocess a given image:
        move channel's dimension to the second position
        normalize pixels to range in [-1, 1]

    Args:
        x (torch.tensor): Input image

    Return:
        x (torch.tensor): Preprocessed (conv usable + normalized) image
    """

    x = x.permute(0, 3, 1, 2)
    x = (x - 127.5) / 127.5

    return x

def deprocess(x):
    """
    Deprocess a given image:
        move channel's dimension to the last position
        normalize pixels to range in [0, 1]

    Args:
        x (torch.tensor): Input image

    Return:
        x (torch.tensor): Deprocessed (plotable) image
    """

    x = x.permute(0, 2, 3, 1)
    x = (x + 1) * 0.5

    return x
