
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
import json

import torch
from fairseq.models.transformer import TransformerModel

import argparse
from torch.serialization import add_safe_globals


import numpy as np
import random
import os

# Ensure PyTorch uses deterministic algorithms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"

class WrappedEncoder(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = TransformerModel.from_pretrained(
            model_path,
            'model.pt'
        )
        device = 'cuda'
        encoder = model.models[0].encoder.eval().to(device)
        self.encoder = encoder

    def forward(self, tokens, lengths):
        output = self.encoder(tokens, lengths)
        # Convert padding mask to int because dlpack doesn't support bool
        return output['encoder_out'][0].permute(1, 0, 2).contiguous().to(tokens.device), output['encoder_padding_mask'][0].to(torch.int64).contiguous().to(tokens.device)


class TritonPythonModel:
    def initialize(self, args):
        add_safe_globals([argparse.Namespace])
        self.model_config = json.loads(args["model_config"])
        model_path = self.model_config["parameters"]["model_path"]["string_value"]

        self.wrapped_encoder = WrappedEncoder(model_path)

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get input tensors and convert to PyTorch tensors
            input_0 = pb_utils.get_input_tensor_by_name(request, "INPUT__0")
            input_1 = pb_utils.get_input_tensor_by_name(request, "INPUT__1")
    
            # Convert Triton tensors to PyTorch and move to CUDA
            tokens = torch.from_numpy(input_0.as_numpy()).to("cuda")
            lengths = torch.from_dlpack(input_1.to_dlpack()).to("cuda")
    
            # Run inference
            encoder_out, encoder_padding_mask = self.wrapped_encoder(
                tokens=tokens,
                lengths=lengths
            )
    
            # Convert outputs to Triton tensors (ensure they stay on CUDA)
            out_tensors = [
                pb_utils.Tensor.from_dlpack("OUTPUT__0", torch.to_dlpack(encoder_out)),
                pb_utils.Tensor.from_dlpack("OUTPUT__1", torch.to_dlpack(encoder_padding_mask))
            ]
            
            responses.append(pb_utils.InferenceResponse(output_tensors=out_tensors))
        return responses


    def finalize(self):
        print("Cleaning up...")