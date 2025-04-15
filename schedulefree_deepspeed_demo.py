#!/usr/bin/env python
# Demo integrating Schedule-Free Optimizer with DeepSpeed

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import deepspeed
import torch.optim as optim
from schedulefree import AdamWScheduleFree  # Import the Schedule-Free optimizer

class SimpleModel(nn.Module):
    """A simple model for demonstration purposes"""
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RandomDataset(Dataset):
    """Generate random data for testing"""
    def __init__(self, size=1000, input_dim=784, num_classes=10):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random input tensor - use float16 to match DeepSpeed's fp16 configuration
        x = torch.randn(self.input_dim, dtype=torch.float16)
        # Generate random target (class index)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y

def add_argument():
    """Setup arguments for DeepSpeed"""
    parser = argparse.ArgumentParser(description='ScheduleFree with DeepSpeed Demo')
    
    # Data and model arguments
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension size')
    
    # DeepSpeed arguments
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    
    # Schedule-Free specific arguments
    parser.add_argument('--lr', type=float, default=0.0025, help='learning rate')
    parser.add_argument('--beta', type=float, default=0.9, help='Schedule-Free momentum parameter')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='weight decay value')
    parser.add_argument('--warmup-steps', type=int, default=100, help='warmup steps')
    
    args = parser.parse_args()
    return args

def create_deepspeed_config(args):
    """Create a DeepSpeed config from command line arguments"""
    config = {
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "zero_allow_untested_optimizer": True,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 16
        },
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
        }
    }
    return config

def create_custom_optimizer(model, args):
    """Create the Schedule-Free optimizer"""
    return AdamWScheduleFree(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta, 0.999),
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps
    )

def main():
    # Parse arguments
    args = add_argument()
    
    # Initialize DeepSpeed distributed environment
    deepspeed.init_distributed()
    
    # Create model
    model = SimpleModel(hidden_dim=args.hidden_dim)
    
    # Create Schedule-Free optimizer
    optimizer = create_custom_optimizer(model, args)
    
    # DeepSpeed initialization
    ds_config = create_deepspeed_config(args)
    
    # Keep reference to the original optimizer for train/eval mode
    sf_optimizer = optimizer
    
    # No LR scheduler needed for Schedule-Free
    lr_scheduler = None
    
    # Model, optimizer, and training data
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config,
        model_parameters=model.parameters()
    )
    
    # Create dataset and dataloader
    train_dataset = RandomDataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank()
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler
    )
    
    # Training loop
    for epoch in range(args.epochs):
        # Set the train_sampler for distributed training
        train_sampler.set_epoch(epoch)
        
        # Training mode
        model_engine.train()
        
        # Important: We need to put Schedule-Free in train mode
        sf_optimizer.train()
        
        epoch_loss = 0.0
        steps = 0
        
        for data, target in train_loader:
            # Forward pass
            data, target = data.to(model_engine.device), target.to(model_engine.device)
            outputs = model_engine(data)
            loss = F.cross_entropy(outputs, target)
            
            # Backward pass
            model_engine.backward(loss)
            model_engine.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            steps += 1
        
        # Print progress
        avg_loss = epoch_loss / steps
        print(f"Epoch: {epoch}, Avg Loss: {avg_loss:.4f}")
        
        # Important: We need to put Schedule-Free in eval mode for validation
        sf_optimizer.eval()
        
        # Save checkpoint
        if torch.distributed.get_rank() == 0:  # Save only on one process
            checkpoint_path = f"checkpoints/epoch_{epoch}"
            client_state = {"epoch": epoch}
            model_engine.save_checkpoint(checkpoint_path, client_state=client_state)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
