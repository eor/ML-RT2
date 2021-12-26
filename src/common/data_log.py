import subprocess
import shlex
import time
from torch.utils.tensorboard import SummaryWriter

class DataLog:
    """
    A class that handles all the data logging and the tensorboard server.
    """

    __instance = None

    @staticmethod
    def getInstance(path=None):
        """ Static access method. """
        if DataLog.__instance is None:
            DataLog(path)
        return DataLog.__instance

    def __init__(self, path):
        if DataLog.__instance is not None:
            raise Exception("This class is a singleton. Please access it using DataLog.getInstance()")
        else:
            DataLog.__instance = self
            if path is not None:
                self.log_dir_path = path
                self.tensorboard = SummaryWriter(log_dir=path)
            else:
                # defaults to: runs/CURRENT_DATETIME_HOSTNAME
                self.tensorboard = SummaryWriter()
            # global_step to record the data for
            self.curr_epoch = 0

    def start_server(self):
        """ Start the tensorboard server in a separate process
        """
        print("\033[96m\033[1m\nStarting tensorboard server\033[0m")
        start_command = 'tensorboard --load_fast=false --logdir='+ self.log_dir_path
        args = shlex.split(start_command)
        self.process = subprocess.Popen(args)
        time.sleep(3)

    def get_tensorboard(self):
        """ Returns the tensorboard instance
        """
        return self.tensorboard

    def log(self, key, value):
        """ log the data as a scalar for the global_step to the tensorboard
        """
        self.tensorboard.add_scalar(key, value, self.curr_epoch)

    def log_losses(self, train_loss, val_loss):
        """ log the train and validation loss as a scalar for the
        global_step to the tensorboard.
        """
        self.tensorboard.add_scalars(f'loss/train_and_validation', {
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, self.curr_epoch)

    def update_data(self):
        """ Update the data to the tensorboard and increment the global_step.
        The data logged after calling this function will be logged for the next
        global_step.
        """
        # increment the global_step value
        self.curr_epoch += 1
        # update the values in the tensorboard
        self.tensorboard.flush()

    def close(self):
        """ Terminates the tensorboard server after saving all the data and
        releasing the local port used for hosting.
        """
        print("\033[96m\033[1m\nTerminating tensorboard server\033[0m")
        if self.process is not None:
            return_code = self.process.terminate()
        time.sleep(3)
        self.tensorboard.close()
