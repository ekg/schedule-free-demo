# DeepSpeed with Schedule-Free Optimizer Demo

This demo shows how to integrate Facebook Research's Schedule-Free optimizer with DeepSpeed for efficient distributed training without learning rate schedules.

## What is Schedule-Free Optimization?

Schedule-Free is an optimization approach that eliminates the need for predefined learning rate schedules while maintaining or exceeding their performance. Unlike traditional schedulers that require specifying the total number of training steps in advance, Schedule-Free adapts automatically through a combination of interpolation and averaging techniques.

Key advantages:
- No need to specify training duration in advance
- Tracks performance comparable to carefully tuned learning rate schedules
- Allows for larger learning rates without divergence
- Compatible with existing optimizers like SGD and Adam

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Schedule-Free from source
pip install git+https://github.com/facebookresearch/schedule_free.git
```

## Running the Demo

The demo can be run on a single GPU or with multiple GPUs using DeepSpeed's distributed training capabilities:

### Single GPU

```bash
deepspeed --num_gpus=1 schedulefree_deepspeed_demo.py --deepspeed --deepspeed_config ds_config.json
```

### Multiple GPUs on a single node

```bash
deepspeed --num_gpus=4 schedulefree_deepspeed_demo.py --deepspeed --deepspeed_config ds_config.json
```

### Multiple nodes

Create a hostfile (e.g., `hostfile.txt`) with the IP addresses and GPU counts:

```
192.168.1.101 slots=8
192.168.1.102 slots=8
```

Then run:

```bash
deepspeed --hostfile=hostfile.txt schedulefree_deepspeed_demo.py --deepspeed --deepspeed_config ds_config.json
```

## Configuration Options

The demo supports various command-line arguments to customize the training:

- `--batch-size`: Batch size per GPU (default: 32)
- `--epochs`: Number of training epochs (default: 10)
- `--hidden-dim`: Hidden dimension size (default: 128)
- `--lr`: Learning rate (default: 0.0025)
- `--beta`: Schedule-Free momentum parameter (default: 0.9)
- `--weight-decay`: Weight decay value (default: 0.01)
- `--warmup-steps`: Number of warmup steps (default: 100)

## Note on Schedule-Free with BatchNorm

If your model uses BatchNorm layers, additional handling is required:

1. Before evaluation, you need to update the BatchNorm statistics for the averaged model:
   ```python
   # Update BatchNorm statistics before evaluation
   model.train()
   optimizer.eval()  # Put Schedule-Free in eval mode but keep model in train mode
   with torch.no_grad():
       for batch in itertools.islice(train_loader, 50):  # Use a few batches to update statistics
           model(batch)
   model.eval()  # Now switch to eval mode for evaluation
   ```

## References

- [The Road Less Scheduled](https://arxiv.org/abs/2405.15682) - Original paper on Schedule-Free optimization
- [Schedule-Free GitHub Repository](https://github.com/facebookresearch/schedule_free)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)

## DeepSpeed Compatibility Note

When using Schedule-Free with DeepSpeed ZeRO optimization, you may need a compatibility wrapper:

```python
# Create a wrapper that inherits from torch.optim.AdamW for better DeepSpeed compatibility
class ScheduleFreeWrapper(optim.AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, warmup_steps=100):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay)
        # Create the actual Schedule-Free optimizer internally
        self.sf_optimizer = AdamWScheduleFree(
            params, lr=lr, betas=betas, weight_decay=weight_decay, warmup_steps=warmup_steps
        )
        
    def step(self, closure=None):
        # Delegate to the Schedule-Free optimizer
        return self.sf_optimizer.step(closure)
        
    def train(self):
        self.sf_optimizer.train()
        
    def eval(self):
        self.sf_optimizer.eval()
```

This wrapper helps DeepSpeed recognize the optimizer as a standard AdamW while still using the Schedule-Free implementation.
