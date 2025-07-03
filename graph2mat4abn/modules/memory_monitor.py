import torch
import matplotlib.pyplot as plt
from collections import defaultdict

class MemoryMonitor:
    def __init__(self):
        self.memory_stats = defaultdict(list)
        self.epoch = 0
        
    def start_epoch(self):
        """Call this at the beginning of each epoch"""
        self.epoch += 1
        self._record_memory()
        
    def end_epoch(self):
        """Call this at the end of each epoch"""
        self._record_memory()
        
    def _record_memory(self):
        """Record current memory stats"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
            self.memory_stats['allocated'].append(allocated)
            self.memory_stats['reserved'].append(reserved)
            self.memory_stats['epoch'].append(self.epoch)
            
    def plot_memory_usage(self, save_path):
        """Plot memory usage over epochs"""
        if not self.memory_stats:
            print("No memory stats recorded")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.memory_stats['epoch'], self.memory_stats['allocated'], label='Allocated Memory (MB)')
        plt.plot(self.memory_stats['epoch'], self.memory_stats['reserved'], label='Reserved Memory (MB)')
        
        plt.xlabel('Epoch')
        plt.ylabel('Memory (MB)')
        plt.title('CUDA Memory Usage Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        
    def print_memory_stats(self):
        """Print current memory statistics"""
        if torch.cuda.is_available():
            print(f"\nEpoch {self.epoch} Memory Stats:")
            print(f"Allocated: {self.memory_stats['allocated'][-1]:.2f} MB")
            print(f"Reserved: {self.memory_stats['reserved'][-1]:.2f} MB")
            print(f"Max allocated: {max(self.memory_stats['allocated']):.2f} MB")
            print(f"Max reserved: {max(self.memory_stats['reserved']):.2f} MB")