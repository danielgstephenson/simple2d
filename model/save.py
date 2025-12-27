import torch
from torch import nn
from torch.export import Dim

def save_checkpoint(
        model: nn.Module,
        target_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        discount: float, 
        horizon: float, 
        path: str
    ):
    checkpoint = { 
        'model_state_dict': model.state_dict(),
        'target_state_dict': target_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'discount': discount,
        'horizon': horizon
    }
    try:
        torch.save(checkpoint, path)
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt detected. Saving checkpoint...')
        torch.save(checkpoint, path)
        print('Checkpoint saved.')
        raise

def save_onnx(model: nn.Module, path: str, device: torch.device):
    # with contextlib.redirect_stdout(io.StringIO()):
    example_input = torch.tensor([[i for i in range(16)]],dtype=torch.float32).to(device)
    example_input_tuple = (example_input,)
    onnx_program = torch.onnx.export(
        model, 
        example_input_tuple, 
        dynamo=True,
        input_names=["state"],
        output_names=["output"],
        dynamic_shapes=[[Dim("batch_size"), Dim.AUTO]],
        verbose=False
    )
    if onnx_program is not None:
        onnx_program.save(path)