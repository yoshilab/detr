import torch

from models import PositionEmbeddingSine
from util.misc import NestedTensor


def main():
    embedding = PositionEmbeddingSine(num_pos_feats=128)
    images = torch.rand((1, 3, 64, 64)).type(torch.FloatTensor)
    masks = torch.rand((1, 64, 64)).type(torch.LongTensor)
    inputs = NestedTensor(images, masks)
    pos = embedding(inputs)
    print(pos.size())


if __name__ == '__main__':
    main()
