import os
import warnings
import numpy as np
import cv2
import shutil

try:
    from ..constants import CWD
except ImportError:
    CWD = os.getcwd()

with open(os.path.join(CWD, "backend_log.txt"), "w") as f:
    pass


def removeFile(file):
    try:
        os.remove(file)
    except Exception:
        print("Failed to remove file!")


def removeFolder(folder):
    try:
        shutil.rmtree(folder)
    except Exception:
        print("Failed to remove file!")


def warnAndLog(message: str):
    warnings.warn(message)
    log("WARN: " + message)


def errorAndLog(message: str):
    log("ERROR: " + message)
    raise os.error("ERROR: " + message)


def printAndLog(message: str, separate=False):
    """
    Prints and logs a message to the log file
    separate, if True, activates the divider
    """
    if separate:
        message = message + "\n" + "---------------------"
    print(message)
    log(message=message)


def log(message: str):
    with open(os.path.join(CWD, "backend_log.txt"), "a") as f:
        f.write(message + "\n")


def bytesToImg(
    image: bytes, width, height, outputWidth: int = None, outputHeight: int = None
) -> np.ndarray:
    frame = np.frombuffer(image, dtype=np.uint8).reshape(height, width, 3)
    if outputHeight and outputWidth:
        frame = cv2.resize(frame, dsize=(100, 100))
    return frame


def get_pytorch_vram() -> int:
    """
    Function that returns the total VRAM amount in MB using PyTorch.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda")
            props = torch.cuda.get_device_properties(device)
            vram_in_mb = props.total_memory // (1024**2)  # Convert bytes to MB
            return vram_in_mb
        else:
            return 0
    except ImportError as e:
        log(str(e))
        return 0
    except Exception as e:
        log(str(e))
        return 0


def checkForPytorchCUDA() -> bool:
    """
    function that checks if the pytorch backend is available
    """
    try:
        import torch
        import torchvision

        if "cu" in torch.__version__:
            return True
        return False
    except ImportError as e:
        log(str(e))
        return False
    except Exception as e:
        log(str(e))


def checkForPytorchROCM() -> bool:
    """
    function that checks if the pytorch backend is available
    """
    try:
        import torch
        import torchvision

        if "rocm" in torch.__version__:
            return True
        return False
    except ImportError as e:
        log(str(e))
        return False
    except Exception as e:
        log(str(e))


def checkForTensorRT() -> bool:
    """
    function that checks if the pytorch backend is available
    """
    try:
        import torch
        import torchvision
        import tensorrt
        import torch_tensorrt

        return True
    except ImportError as e:
        log(str(e))
        return False
    except Exception as e:
        log(str(e))


def checkForGMFSS() -> bool:
    try:
        import torch
        import torchvision
        import cupy
    except ImportError as e:
        log(str(e))
        return False
    if cupy.cuda.get_cuda_path() == None:
        return False
    return True


def check_bfloat16_support() -> bool:
    """
    Function that checks if the torch backend supports bfloat16
    """
    import torch

    try:
        x = torch.tensor([1.0], dtype=torch.float16).cuda()
        return True
    except RuntimeError:
        return False


def checkForDirectMLHalfPrecisionSupport() -> bool:
    """
    Function that checks if the onnxruntime DirectML backend supports half precision
    """
    try:
        import onnxruntime as ort
        import numpy as np
        import onnx
        from onnx import helper, TensorProto

        # Check if DirectML execution provider is available
        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            # Create a dummy model with half precision input
            input_shape = [1, 3, 224, 224]  # Example input shape
            input_tensor = helper.make_tensor_value_info(
                "input", TensorProto.FLOAT16, input_shape
            )
            output_tensor = helper.make_tensor_value_info(
                "output", TensorProto.FLOAT16, input_shape
            )
            node = helper.make_node("Identity", ["input"], ["output"])
            graph = helper.make_graph([node], "test", [input_tensor], [output_tensor])

            # Create the model
            model = helper.make_model(graph, producer_name="test_model")

            # Add opset version
            opset = helper.make_operatorsetid(
                "", 13
            )  # Use opset version 13 or any other appropriate version
            model.opset_import.extend([opset])

            # Set the IR version
            model.ir_version = onnx.IR_VERSION

            # Create an inference session with DirectML execution provider
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            session_options.add_session_config_entry("session.use_dml", "1")
            session = ort.InferenceSession(model.SerializeToString(), session_options)

            # Check if the model can be run with half precision input
            input_data = np.random.randn(*input_shape).astype(np.float16)
            outputs = session.run(None, {"input": input_data})

            return True
        else:
            log("DirectML execution provider not available")
            return False
    except ImportError as e:
        log(str(e))
        return False
    except Exception as e:
        log(str(e))
        return False


def checkForDirectML() -> bool:
    """
    Function that checks if the onnxruntime DirectML backend is available
    """
    try:
        import onnxruntime as ort
        import onnx
        import onnxconverter_common

        # Check if DirectML execution provider is available
        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            return True
        else:
            log("DirectML execution provider not available")
            return False
    except ImportError as e:
        log(str(e))
        return False
    except Exception as e:
        log(str(e))
        return False


def checkForNCNN() -> bool:
    """
    function that checks if the pytorch backend is available
    """
    try:
        from rife_ncnn_vulkan_python import Rife
        import ncnn

        try:
            from upscale_ncnn_py import UPSCALE
        except Exception:
            printAndLog(
                "Warning: Cannot import upscale_ncnn, falling back to default ncnn processing. (Please install vcredlist on your computer to fix this!)"
            )
        return True
    except ImportError as e:
        log(str(e))
        return False
    except Exception as e:
        log(str(e))


def get_gpus_torch():
    """
    Function that returns a list of available GPU names using PyTorch.
    """
    devices = []
    try:
        import torch
        if torch.cuda.is_available():
            for dev_index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(dev_index)
                devices.append(props.name)
        if not devices:
            devices.append("CPU")
    except ImportError as e:
        log(str(e))
        devices.append("CPU")
    except Exception as e:
        log(str(e))
        devices.append("CPU")
    return devices

def get_gpus_ncnn():
    devices = []
    try:
        import ncnn
        gpu_count = ncnn.get_gpu_count()
        if gpu_count < 1:
            return "CPU"
        for i in range(gpu_count):
            device = ncnn.get_gpu_device(0)
            gpu_info = device.info()  
            devices.append(gpu_info.device_name())
        return devices
    except Exception as e:
        log(str(e))
        return "Unable to get NCNN GPU"

if __name__ == "__main__":
    print(get_gpus_ncnn())
    print(get_gpus_torch())
