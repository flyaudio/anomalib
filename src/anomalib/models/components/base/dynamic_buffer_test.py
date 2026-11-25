from dynamic_buffer import DynamicBufferMixin
import torch
from torch import nn


def test_DynamicBufferMixin():
    class MyModule(DynamicBufferMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.ones(3)) # Original buffer shape is (3,)

    module = MyModule()
    print("before", module.buffer)
    # Load state dict with different buffer shape (5,)
    new_state = {"buffer": torch.ones(5)}
    module.load_state_dict(new_state)
    # Buffer is automatically resized
    print("after:", module.buffer)


if __name__ == "__main__":
    test_DynamicBufferMixin()
