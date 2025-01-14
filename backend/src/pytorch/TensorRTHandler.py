"""
MIT License

Copyright (c) 2024 TNTwise

cPermission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import os
from ..utils.Util import suppress_stdout_stderr
with suppress_stdout_stderr():
    import torch
    import torch_tensorrt
    import tensorrt as trt
    from torch._decomp import get_decompositions
    from torch._export.converter import TS2EPConverter
    from torch.export.exported_program import ExportedProgram

def torchscript_to_dynamo(
            model: torch.nn.Module, example_inputs: list[torch.Tensor]
        ) -> ExportedProgram:
            """Converts a TorchScript module to a Dynamo program."""
            module = torch.jit.trace(model, example_inputs)
            exported_program = TS2EPConverter(
                module, sample_args=tuple(example_inputs), sample_kwargs=None
            ).convert()
            del module
            torch.cuda.empty_cache()
            return exported_program

def nnmodule_to_dynamo(
    model: torch.nn.Module, example_inputs: list[torch.Tensor]
) -> ExportedProgram:
    """Converts a nn.Module to a Dynamo program."""
    return torch.export.export(
        model, tuple(example_inputs), dynamic_shapes=None
    )

"""onnx_support = True
try:
    import onnx
except ImportError:
    onnx_support = False"""


class TorchTensorRTHandler:
    """
    Args:
        dynamo_export_format (str): The export format to use when exporting models using Dynamo. Defaults to "nn2exportedprogram".
        trt_workspace_size (int): The workspace size to use when compiling models using TensorRT. Defaults to 0.
        max_aux_streams (int | None): The maximum number of auxiliary streams to use when compiling models using TensorRT. Defaults to None.
        trt_optimization_level (int): The optimization level to use when compiling models using TensorRT. Defaults to 3.
        debug (bool): Whether to enable debugging when compiling models using TensorRT. Defaults to False.
        static_shape (bool): Whether to use static shape when compiling models using TensorRT. Defaults

        dynamo_export_format (str): nn2exportedprogram, torchscript2exportedprogram or fallback, which tries nn2exportedprogram first, and then torchscrip2exportedprogram if nn2exportedprogram fails, torchscript2exportedprogram uses torchscript as an intermediatory as some issues occur when using dynamo with torch.export.export

        multi precision engines seem to not like torchscript2exportedprogram,
        or maybe its just the model not playing nice with explicit_typing,
        either way, forcing one precision helps with speed in some cases.
    """

    def __init__(
        self,
        export_format: str = "dynamo",
        dynamo_export_format: str = "nn2exportedprogram",
        multi_precision_engine: bool = True,
        trt_workspace_size: int = 0,
        max_aux_streams: int | None = None,
        trt_optimization_level: int = 3,
        debug: bool = False,
        static_shape: bool = True,
    ):
        self.tensorrt_version = trt.__version__  # can just grab version from here instead of importing trt and torch trt in all related files
        self.torch_tensorrt_version = torch_tensorrt.__version__

        self.export_format = export_format
        self.dynamo_export_format = dynamo_export_format
        self.trt_workspace_size = trt_workspace_size
        self.max_aux_streams = max_aux_streams
        self.optimization_level = trt_optimization_level
        self.debug = debug
        self.static_shape = static_shape  # Unused for now
        self.multi_precision_engine = multi_precision_engine

    def prepare_inputs(
        self, example_inputs: list[torch.Tensor]
    ) -> list[torch_tensorrt.Input]:
        """Prepares input specifications for TensorRT."""
        return [
            torch_tensorrt.Input(shape=input.shape, dtype=input.dtype)
            for input in example_inputs
        ]
    
    @torch.inference_mode()
    def dynamo_multi_precision_export(self, exported_program, example_inputs, device):
        
            return torch_tensorrt.dynamo.compile(
                exported_program,
                tuple(self.prepare_inputs(example_inputs)),
                device=device,
                use_explicit_typing=True,
                debug=self.debug,
                num_avg_timing_iters=4,
                workspace_size=self.trt_workspace_size,
                min_block_size=1,
                max_aux_streams=self.max_aux_streams,
                optimization_level=self.optimization_level,
            )
    
    def dynamo_export(self, exported_program, example_inputs, device, dtype):
            return torch_tensorrt.dynamo.compile(
                exported_program,
                tuple(self.prepare_inputs(example_inputs)),
                device=device,
                enabled_precisions={dtype},
                debug=self.debug,
                num_avg_timing_iters=4,
                workspace_size=self.trt_workspace_size,
                min_block_size=1,
                max_aux_streams=self.max_aux_streams,
                optimization_level=self.optimization_level,
            )

    def grid_sample_decomp(self, exported_program):
        return exported_program.run_decompositions(
            get_decompositions([torch.ops.aten.grid_sampler_2d])
        )

    def export_using_dynamo(
        self,
        model: torch.nn.Module,
        example_inputs: list[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        trt_engine_path: str,
    ):

        """Exports a model using TensorRT Dynamo."""
        if self.dynamo_export_format == "nn2exportedprogram":
            exported_program = nnmodule_to_dynamo(model, example_inputs)
        elif self.dynamo_export_format == "torchscript2exportedprogram":
            exported_program = torchscript_to_dynamo(model, example_inputs)
        elif self.dynamo_export_format == "fallback":
            try:
                exported_program = nnmodule_to_dynamo(model, example_inputs)
            except Exception as e:
                print(
                    "Failed to export using nn2exportedprogram. Falling back to torchscript2exportedprogram...",
                    file=sys.stderr,
                )
                exported_program = torchscript_to_dynamo(model, example_inputs)
        else:
            raise ValueError(f"Unsupported export format: {self.dynamo_export_format}")

        torch.cuda.empty_cache()

        exported_program = self.grid_sample_decomp(exported_program)
        
        if self.multi_precision_engine:
            model_trt = self.dynamo_multi_precision_export(exported_program, example_inputs, device)
        else:
            model_trt = self.dynamo_export(exported_program, example_inputs, device, dtype)

        torch_tensorrt.save(
            model_trt,
            trt_engine_path,
            output_format="torchscript",
            inputs=tuple(example_inputs),
        )
        torch.cuda.empty_cache()

    def export_torchscript_model(
        self,
        model: torch.nn.Module,
        example_inputs: list[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        trt_engine_path: str,
    ):
        """Exports a model using TorchScript."""

        model.to(device=device, dtype=dtype)
        module = torch.jit.trace(model, example_inputs)
        torch.cuda.empty_cache()
        del model

        module_trt = torch_tensorrt.compile(
            module,
            ir="ts",
            inputs=example_inputs,
            enabled_precisions={dtype},
            device=torch_tensorrt.Device(gpu_id=0),
            workspace_size=self.trt_workspace_size,
            truncate_long_and_double=True,
            min_block_size=1,
        )
        torch.jit.save(module_trt, trt_engine_path)

    def build_engine(
        self,
        model: torch.nn.Module,
        dtype: torch.dtype,
        device: torch.device,
        example_inputs: list[torch.Tensor],
        trt_engine_path: str,
    ):
        torch.cuda.empty_cache()
        """Builds a TensorRT engine from the provided model."""
        print(
            f"Building TensorRT engine {os.path.basename(trt_engine_path)}. This may take a while...",
            file=sys.stderr,
        )
        if self.export_format == "dynamo":
            if self.debug:
                self.export_using_dynamo(
                    model, example_inputs, device, dtype, trt_engine_path
                )
            else:
                 with suppress_stdout_stderr():
                    self.export_using_dynamo(
                        model, example_inputs, device, dtype, trt_engine_path
                    )
        elif self.export_format == "torchscript":
            self.export_torchscript_model(
                model, example_inputs, device, dtype, trt_engine_path
            )
        else:
            raise ValueError(f"Unsupported export format: {self.export_format}")
        
        torch.cuda.empty_cache()

    def load_engine(self, trt_engine_path: str) -> torch.jit.ScriptModule:
        """Loads a TensorRT engine from the specified path."""
        print(f"Loading TensorRT engine from {trt_engine_path}.", file=sys.stderr)
        return torch.jit.load(trt_engine_path).eval()


"""
class TensorRTHandler:
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    def __init__(
        self,
        trt_workspace_size: int = 0,
        max_aux_streams: int | None = None,
        trt_optimization_level: int = 3,
        static_shape: bool = True,
    ):
        self.tensorrt_version = trt.__version__
        self.trt_workspace_size = trt_workspace_size
        self.max_aux_streams = max_aux_streams
        self.optimization_level = trt_optimization_level
        self.static_shape = static_shape
    
    def export_onnx(
            self,
            model,
            dtype,
            device,
            example_inputs: list[torch.Tensor]
            ):
        example_inputs = [input.to(device=device, dtype=dtype) for input in example_inputs]
        model.to(device=device, dtype=dtype)
        with BytesIO() as f:
            torch.onnx.export(
                model,
                tuple(example_inputs),
                f,
                verbose=True,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                #dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}, dealing with static for testing currently
            )
            f.seek(0)
            return f.read()

    def build_tensorrt_engine(self, onnx_model, trt_engine_path):
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        parser.parse(onnx_model)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
        serialized_engine = builder.build_serialized_network(network, config)
        with open(trt_engine_path, "wb") as f:
            f.write(serialized_engine)

    def load_engine(self, trt_engine_path: str):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(trt_engine_path, "rb") as f:
            serialized_engine = f.read()
        engine =  runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()

    def build_engine(
        self,
        model: torch.nn.Module,
        dtype: torch.dtype,
        device: torch.device,
        example_inputs: list[torch.Tensor],
        trt_engine_path: str,
    ):
        onnx_model = self.export_onnx(model=model, example_inputs=example_inputs, dtype=dtype, device=device)
        engine = self.build_tensorrt_engine(onnx_model, trt_engine_path)

    def __call__(self): # inference here
        pass

if __name__ == '__main__':
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    example_inputs = [torch.randn(10)]
    trt_engine_path = "model.engine"
    handler = TensorRTHandler()
    handler.build_engine(model, torch.float32, torch.device("cuda"), example_inputs, trt_engine_path)
    engine = handler.load_engine(trt_engine_path)"""
