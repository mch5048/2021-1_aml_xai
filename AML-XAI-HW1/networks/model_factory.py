import torch
class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(dataset, trainer, task_info):
        
        if dataset == 'CIFAR100':
            
            import networks.network as net
            return net.conv_net(task_info)
        
        elif dataset == 'MNIST':
            
            import networks.network as net
            return net.MLP(task_info)