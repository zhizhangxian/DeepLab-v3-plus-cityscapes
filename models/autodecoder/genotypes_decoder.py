from collections import namedtuple

Genotype = namedtuple('Genotype', 'cell cell_concat')

PRIMITIVES = [
    'identity',
    'conv3x3',
    'conv5x5',
    'conv3x3x2',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'bottle_3x3',
    'bottle_5x5',
]