import copy
from arguments import get_args
args = get_args()
import torch

class TrainerFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(myModel, args, optimizer, evaluator, task_info):
        
        if args.trainer == 'ewc':
            import trainer.ewc as trainer
        elif args.trainer == 'vanilla':
            import trainer.vanilla as trainer
        elif args.trainer == 'l2':
            import trainer.l2 as trainer
        
        return trainer.Trainer(myModel, args, optimizer, evaluator, task_info)
    
class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''

    def __init__(self, model, args, optimizer, evaluator, task_info):
        
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.evaluator=evaluator
        self.task_info=task_info
        self.model_fixed = copy.deepcopy(self.model)
        for param in self.model_fixed.parameters():
            param.requires_grad = False
        self.current_lr = args.lr
        self.ce=torch.nn.CrossEntropyLoss()
        self.model_single = copy.deepcopy(self.model)
        self.optimizer_single = None
        
        
    def update_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from %0.4f to %0.4f"%(self.current_lr,
                                                                        self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]

        
    def setup_training(self, lr):
        
        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.4f"%lr)
            param_group['lr'] = lr
            self.current_lr = lr

    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False