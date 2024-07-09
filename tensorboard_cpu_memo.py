import psutil
import os
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter

class MemoryMonitorCallback(TrainerCallback):
    def __init__(self, log_dir=None):
        self.process = psutil.Process(os.getpid())
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def on_step_end(self, args, state, control, **kwargs):
        memory_info = self.process.memory_info()
        self.writer.add_scalar('Memory/CPU', memory_info.rss / (1024 * 1024), self.step)
        self.step += 1

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()
