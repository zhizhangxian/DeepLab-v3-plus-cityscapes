from collections import namedtuple

Genotype = namedtuple('Genotype', 'cell cell_concat')

PRIMITIVES = [
    'identity',
    'conv3x3',
    'conv5x5',
    'conv3x3x2',
    'conv3x3_conv5x5',
    'conv3x3_conv5x5',
    'conv3x3_conv5x5',
]