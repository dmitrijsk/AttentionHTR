"""
A modified version of early-stopping-pytorch repository https://github.com/Bjarten/early-stopping-pytorch
License: https://github.com/Bjarten/early-stopping-pytorch/blob/master/LICENSE
"""

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model, valid_log, valid_log_fname):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            
            valid_log = valid_log + " " + f"Current score: {score}. Best score: {self.best_score}. Counter: [{self.counter}/{self.patience}]"
            self.log(valid_log, valid_log_fname)
        
            self.save_checkpoint(val_loss, model, valid_log_fname)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                
            valid_log = valid_log + " " + f"Current score: {score}. Best score: {self.best_score}. Counter: [{self.counter}/{self.patience}]"
            self.log(valid_log, valid_log_fname)
        
        else:
            self.best_score = score
            
            valid_log = valid_log + " " + f"Current score: {score}. Best score: {self.best_score}. Counter: [0/{self.patience}]"
            self.log(valid_log, valid_log_fname)
        
            self.save_checkpoint(val_loss, model, valid_log_fname)
            self.counter = 0


    def log(self, valid_log, valid_log_fname):
        with open(valid_log_fname, 'a') as valid_file:
            print(valid_log)
            valid_file.write(valid_log + "\n")
    
    def save_checkpoint(self, val_loss, model, valid_log_fname):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            log_text = f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            # self.trace_func(log_text)
            self.log(log_text, valid_log_fname)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
