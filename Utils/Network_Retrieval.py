from Models import VGGModels

ERROR_MESSAGE = 'Invalid network name'


def get_network(name: str):
    net_name = name.lower()
    if net_name.startswith('vgg'):
        depth = int(net_name.replace('vgg', ''))
        if depth != 11 or depth != 13 or depth != 16 or depth != 19:
            print(ERROR_MESSAGE)
        else:
            return VGGModels.VGG(depth=depth)
    else:
        print(ERROR_MESSAGE)
