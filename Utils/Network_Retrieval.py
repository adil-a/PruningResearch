from Models import VGGModels
from typing import List, Union

ERROR_MESSAGE = 'Invalid network name'


def get_network(name: str, dataset: str, config: List[Union[int, str]]):
    ds = dataset.lower()
    net_name = name.lower()
    if net_name.startswith('vgg'):
        depth = int(net_name.replace('vgg', ''))
        if depth != 11 and depth != 13 and depth != 16 and depth != 19:
            print(ERROR_MESSAGE)
        else:
            return VGGModels.VGG(depth=depth, dataset=ds, cfg=config)
    else:
        print(ERROR_MESSAGE)
