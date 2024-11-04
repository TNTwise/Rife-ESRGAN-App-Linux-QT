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
        trt_max_aux_streams: int | None = None,
        trt_optimization_level: int = 5,
        trt_cache_dir: str = modelsDirectory(),
        trt_debug: bool = False,
        trt_static_shape: bool = True,
    ):
        self.export_format = export_format
        self.trt_workspace_size = trt_workspace_size
        self.trt_max_aux_streams = trt_max_aux_streams
        self.trt_optimization_level = trt_optimization_level
        self.trt_cache_dir = trt_cache_dir
        self.trt_debug = trt_debug
        self.trt_static_shape = trt_static_shape  # unused for now

    def prepare_inputs(self, example_inputs):
        inputs = []
        for input in example_inputs:
            inputs.append(torch_tensorrt.Input(shape=input.shape, dtype=input.dtype))
        return inputs

    def build_engine(
        self,
        model: torch.nn.Module,
        dtype: torch.dtype,
        device: torch.device,
        example_inputs: list[torch.Tensor],
        trt_engine_path: str,
    ):
        if self.export_format == "dynamo":
            model.to(device=device,dtype=dtype)
            exported_program = torch.export.export(
                model,
                tuple(example_inputs),
                dynamic_shapes=None,
            )
            exported_program = exported_program.run_decompositions(
                                get_decompositions([torch.ops.aten.grid_sampler_2d])
                            ) # this is a workaround for a bug in tensorrt where grid_sample has a bad output
            model = torch_tensorrt.dynamo.compile(
                exported_program,
                tuple(self.prepare_inputs(example_inputs)),
                device=device,
                use_explicit_typing=True, # this allows for multi-precision engines
                debug=self.trt_debug,
                num_avg_timing_iters=4,
                workspace_size=self.trt_workspace_size,
                min_block_size=1,
                max_aux_streams=self.trt_max_aux_streams,
                optimization_level=self.trt_optimization_level,
            )
            torch_tensorrt.save(
                model,
                trt_engine_path,
                output_format="torchscript",
                inputs=tuple(example_inputs),
            )
        
        # this is better for upscaling because it exports with jit, but does not have an optimization level
        if self.export_format == "torchscript":
            dummy_input_cpu_fp32 = [
                torch.zeros(
                    (1, 3, 32, 32),
                    dtype=torch.float32,
                    device="cpu",
                )
            ]

            module = torch.jit.trace(model.float().cpu(), dummy_input_cpu_fp32)
            module.to(device=device, dtype=dtype)
            module = torch_tensorrt.compile(
                module,
                ir="ts",
                inputs=example_inputs,
                enabled_precisions={dtype},
                device=torch_tensorrt.Device(gpu_id=0),
                workspace_size=self.trt_workspace_size,
                truncate_long_and_double=True,
                min_block_size=1,
            )
            torch.jit.save(module, trt_engine_path)
            
    def load_engine(self, trt_engine_path: str):
        if self.export_format == "dynamo" or self.export_format == "torchscript":
            return torch.jit.load(trt_engine_path).eval()
            
