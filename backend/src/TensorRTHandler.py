import tensorrt
import torch
import torch_tensorrt
from .Util import modelsDirectory
from torch._decomp import get_decompositions


class TorchTensorRTHandler:
    def __init__(
        self,
        export_format: str = "dynamo",
        trt_workspace_size: int = 0,
        max_aux_streams: int | None = None,
        trt_optimization_level: int = 3,
        trt_cache_dir: str = modelsDirectory(),
        debug: bool = False,
        static_shape: bool = True,
    ):
        self.export_format = export_format
        self.trt_workspace_size = trt_workspace_size
        self.max_aux_streams = max_aux_streams
        self.optimization_level = trt_optimization_level
        self.cache_dir = trt_cache_dir
        self.debug = debug
        self.static_shape = static_shape  # Unused for now

    def prepare_inputs(self, example_inputs: list[torch.Tensor]) -> list[torch_tensorrt.Input]:
        """Prepares input specifications for TensorRT."""
        return [torch_tensorrt.Input(shape=input.shape, dtype=input.dtype) for input in example_inputs]

    def export_dynamo_model(self, model: torch.nn.Module, example_inputs: list[torch.Tensor], device: torch.device, dtype: torch.dtype, trt_engine_path: str):
        """Exports a model using TensorRT Dynamo."""
        model.to(device=device, dtype=dtype)
        exported_program = torch.export.export(model, tuple(example_inputs), dynamic_shapes=None)
        exported_program = exported_program.run_decompositions(get_decompositions([torch.ops.aten.grid_sampler_2d]))

        model_trt = torch_tensorrt.dynamo.compile(
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
        
        torch_tensorrt.save(model_trt, trt_engine_path, output_format="torchscript", inputs=tuple(example_inputs))

    def export_torchscript_model(self, model: torch.nn.Module, example_inputs: list[torch.Tensor], device: torch.device, dtype: torch.dtype, trt_engine_path: str):
        """Exports a model using TorchScript."""
        dummy_input_cpu_fp32 = [torch.zeros((1, 3, 32, 32), dtype=torch.float32, device="cpu")]
        module = torch.jit.trace(model.float().cpu(), dummy_input_cpu_fp32)
        module.to(device=device, dtype=dtype)

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
        """Builds a TensorRT engine from the provided model."""
        if self.export_format == "dynamo":
            self.export_dynamo_model(model, example_inputs, device, dtype, trt_engine_path)
        elif self.export_format == "torchscript":
            self.export_torchscript_model(model, example_inputs, device, dtype, trt_engine_path)
        else:
            raise ValueError(f"Unsupported export format: {self.export_format}")

    def load_engine(self, trt_engine_path: str) -> torch.jit.ScriptModule:
        """Loads a TensorRT engine from the specified path."""
        return torch.jit.load(trt_engine_path).eval()