import torch

##########################################################
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_float32_matmul_precision("medium")
torch.set_grad_enabled(False)
grid_cache = {}
batch_cache = {}
##########################################################


@torch.inference_mode()
def forward(tenIn, tenFlow):
    """
    Forward pass of the Softsplat function.

    Parameters:
        tenIn (torch.Tensor): Input tensor of shape [N, C, H, W]
        tenFlow (torch.Tensor): Flow tensor of shape [N, 2, H, W]

    Returns:
        torch.Tensor: Output tensor of shape [N, C, H, W]
    """
    N, C, H, W = tenIn.size()
    device = tenIn.device
    origdtype = tenIn.dtype

    # Initialize output tensor
    tenOut = torch.zeros_like(tenIn)

    key = (H, W, device, origdtype)
    if key not in grid_cache:
        # Create meshgrid of pixel coordinates
        gridY, gridX = torch.meshgrid(
            torch.arange(H, device=device, dtype=origdtype),
            torch.arange(W, device=device, dtype=origdtype),
            indexing="ij",
        )  # [H, W]
        # Cache the grids
        grid_cache[key] = (
            gridY.unsqueeze(0).unsqueeze(0).expand(N, 1, H, W),
            gridX.unsqueeze(0).unsqueeze(0).expand(N, 1, H, W),
        )

    if key not in batch_cache:
        batch_cache[key] = (
            torch.arange(N, device=device).view(N, 1, 1).expand(N, H, W).reshape(-1)
        )

    gridY, gridX = grid_cache[key]
    batch_indices = batch_cache[key]

    # Compute fltX and fltY
    fltX = gridX + tenFlow[:, 0:1, :, :]
    fltY = gridY + tenFlow[:, 1:2, :, :]

    # Flatten variables
    fltX_flat = fltX.reshape(-1)
    fltY_flat = fltY.reshape(-1)
    tenIn_flat = tenIn.permute(0, 2, 3, 1).reshape(-1, C)

    # Finite mask
    finite_mask = torch.isfinite(fltX_flat) & torch.isfinite(fltY_flat)
    if not finite_mask.any():
        return tenOut

    fltX_flat = fltX_flat[finite_mask]
    fltY_flat = fltY_flat[finite_mask]
    tenIn_flat = tenIn_flat[finite_mask]
    batch_indices = batch_indices[finite_mask]

    # Compute integer positions
    intNW_X = torch.floor(fltX_flat).to(dtype=origdtype)
    intNW_Y = torch.floor(fltY_flat).to(dtype=origdtype)
    intNE_X = intNW_X + 1
    intNE_Y = intNW_Y
    intSW_X = intNW_X
    intSW_Y = intNW_Y + 1
    intSE_X = intNW_X + 1
    intSE_Y = intNW_Y + 1

    # Compute weights
    fltNW = (intSE_X - fltX_flat) * (intSE_Y - fltY_flat)
    fltNE = (fltX_flat - intSW_X) * (intSW_Y - fltY_flat)
    fltSW = (intNE_X - fltX_flat) * (fltY_flat - intNE_Y)
    fltSE = (fltX_flat - intNW_X) * (fltY_flat - intNW_Y)

    # Prepare output tensor flat
    tenOut_flat = tenOut.permute(0, 2, 3, 1).reshape(-1, C)

    positions_all_x = torch.cat([intNW_X, intNE_X, intSW_X, intSE_X], dim=0)
    positions_all_y = torch.cat([intNW_Y, intNE_Y, intSW_Y, intSE_Y], dim=0)
    weights_all = torch.cat([fltNW, fltNE, fltSW, fltSE], dim=0)
    batch_all = torch.cat(
        [batch_indices, batch_indices, batch_indices, batch_indices], dim=0
    )

    tenIn_flat_corners = torch.cat(
        [tenIn_flat, tenIn_flat, tenIn_flat, tenIn_flat], dim=0
    )

    valid_mask_all = (
        (positions_all_x >= 0)
        & (positions_all_x < W)
        & (positions_all_y >= 0)
        & (positions_all_y < H)
    )
    positions_all_x = positions_all_x[valid_mask_all]
    positions_all_y = positions_all_y[valid_mask_all]
    weights_all = weights_all[valid_mask_all]
    batch_all = batch_all[valid_mask_all]
    vals = tenIn_flat_corners[valid_mask_all] * weights_all.unsqueeze(1)

    idx_nhw = (
        batch_all.to(dtype=torch.int32) * H * W
        + positions_all_y.to(dtype=torch.int32) * W
        + positions_all_x.to(dtype=torch.int32)
    )

    tenOut_flat.index_add_(0, idx_nhw, vals)

    # Reshape tenOut back to [N, C, H, W]
    tenOut = tenOut_flat.view(N, H, W, C).permute(0, 3, 1, 2)

    return tenOut


@torch.inference_mode()
def softsplat(
    tenIn: torch.Tensor, tenFlow: torch.Tensor, tenMetric: torch.Tensor, strMode: str
):
    mode_parts = strMode.split("-")
    mode_main = mode_parts[0]
    mode_sub = mode_parts[1] if len(mode_parts) > 1 else None

    assert mode_main in ["sum", "avg", "linear", "soft"]
    if mode_main in ["sum", "avg"]:
        assert tenMetric is None
    if mode_main in ["linear", "soft"]:
        assert tenMetric is not None

    mode_to_operation = {
        "avg": lambda: torch.cat(
            [
                tenIn,
                tenIn.new_ones([tenIn.shape[0], 1, tenIn.shape[2], tenIn.shape[3]]),
            ],
            1,
        ),
        "linear": lambda: torch.cat([tenIn * tenMetric, tenMetric], 1),
        "soft": lambda: torch.cat([tenIn * tenMetric.exp(), tenMetric.exp()], 1),
    }

    if mode_main in mode_to_operation:
        tenIn = mode_to_operation[mode_main]()

    tenOut = forward(tenIn, tenFlow)

    if mode_main in ["avg", "linear", "soft"]:
        tenNormalize = tenOut[:, -1:, :, :]

        normalize_modes = {
            None: lambda x: x + 0.0000001,
            "addeps": lambda x: x + 0.0000001,
            "zeroeps": lambda x: torch.where(
                x == 0.0, torch.tensor(1.0, device=x.device, dtype=x.dtype), x
            ),
            "clipeps": lambda x: x.clip(0.0000001, None),
        }

        if mode_sub in normalize_modes:
            tenNormalize = normalize_modes[mode_sub](tenNormalize)

        tenOut = tenOut[:, :-1, :, :] / tenNormalize

    return tenOut


class SoftSplat(torch.nn.Module):
    def __init__(self, mode: str):
        super(SoftSplat, self).__init__()
        self.mode = mode
        mode_parts = mode.split("-")
        mode_main = mode_parts[0]
        self.mode_sub = mode_parts[1] if len(mode_parts) > 1 else None
        self.op = None
        self.normalize = False
        match mode:
            case "avg":
                self.op = self.avg
            case "linear":
                self.op = self.linear
            case "soft":
                self.op = self.soft
        
        if mode_main in ["avg", "linear", "soft"]:
            self.normalize = True

    @torch.inference_mode()
    def norm(self, tenOut: torch.Tensor):
        if self.normalize:
            tenNormalize = tenOut[:, -1:, :, :]

            self.normalize_modes = {
                None: lambda x: x + 0.0000001,
                "addeps": lambda x: x + 0.0000001,
                "zeroeps": lambda x: torch.where(
                    x == 0.0, torch.tensor(1.0, device=x.device,dtype=x.dtype), x
                ),
                "clipeps": lambda x: x.clip(0.0000001, None),
            }

            if self.mode_sub in self.normalize_modes:
                tenNormalize = self.normalize_modes[self.mode_sub](tenNormalize)

            tenOut = tenOut[:, :-1, :, :] / tenNormalize
        return tenOut

    @staticmethod
    @torch.inference_mode()
    def avg(tenIn: torch.Tensor):
        return torch.cat(
                [
                    tenIn,
                    tenIn.new_ones([tenIn.shape[0], 1, tenIn.shape[2], tenIn.shape[3]]),
                ],
                1,
            ),
    @staticmethod
    @torch.inference_mode()
    def linear(tenIn: torch.Tensor, tenMetric: torch.Tensor):
        return torch.cat([tenIn * tenMetric, tenMetric], 1)
    
    @staticmethod
    @torch.inference_mode()
    def soft(tenIn: torch.Tensor, tenMetric: torch.Tensor):
        return torch.cat([tenIn * tenMetric.exp(), tenMetric.exp()], 1)

    @torch.inference_mode()
    def forward(self, tenIn, tenFlow, tenMetric, strMode="soft"):
        if self.op is not None:
            tenIn = self.op(tenIn, tenMetric)
        """
        Forward pass of the Softsplat function.

        Parameters:
            tenIn (torch.Tensor): Input tensor of shape [N, C, H, W]
            tenFlow (torch.Tensor): Flow tensor of shape [N, 2, H, W]

        Returns:
            torch.Tensor: Output tensor of shape [N, C, H, W]
        """
        N, C, H, W = tenIn.size()
        device = tenIn.device
        origdtype = tenIn.dtype

        # Initialize output tensor
        tenOut = torch.zeros_like(tenIn)
        
        # Create meshgrid of pixel coordinates
        gridY, gridX = torch.meshgrid(
            torch.arange(H, device=device, dtype=origdtype),
            torch.arange(W, device=device, dtype=origdtype),
            indexing='ij'
        )  # [H, W]
        # Cache the grids
        gridY,gridX = (gridY.unsqueeze(0).unsqueeze(0).expand(N, 1, H, W), gridX.unsqueeze(0).unsqueeze(0).expand(N, 1, H, W))
        

        batch_indices = torch.arange(N, device=device).view(N, 1, 1).expand(N, H, W).reshape(-1)
        
            

        # Compute fltX and fltY
        fltX = gridX + tenFlow[:, 0:1, :, :]
        fltY = gridY + tenFlow[:, 1:2, :, :]

        # Flatten variables
        fltX_flat = fltX.reshape(-1)
        fltY_flat = fltY.reshape(-1)
        tenIn_flat = tenIn.permute(0, 2, 3, 1).reshape(-1, C)

        

        # Finite mask
        finite_mask = torch.isfinite(fltX_flat) & torch.isfinite(fltY_flat)
        

        fltX_flat = fltX_flat[finite_mask]
        fltY_flat = fltY_flat[finite_mask]
        tenIn_flat = tenIn_flat[finite_mask]
        batch_indices = batch_indices[finite_mask]

        # Compute integer positions
        intNW_X = torch.floor(fltX_flat).to(dtype=origdtype)
        intNW_Y = torch.floor(fltY_flat).to(dtype=origdtype)
        intNE_X = intNW_X + 1
        intNE_Y = intNW_Y
        intSW_X = intNW_X
        intSW_Y = intNW_Y + 1
        intSE_X = intNW_X + 1
        intSE_Y = intNW_Y + 1

        # Compute weights
        fltNW = (intSE_X - fltX_flat) * (intSE_Y - fltY_flat)
        fltNE = (fltX_flat - intSW_X) * (intSW_Y - fltY_flat)
        fltSW = (intNE_X - fltX_flat) * (fltY_flat - intNE_Y)
        fltSE = (fltX_flat - intNW_X) * (fltY_flat - intNW_Y)

        # Prepare output tensor flat
        tenOut_flat = tenOut.permute(0, 2, 3, 1).reshape(-1, C)

        positions_all_x = torch.cat([intNW_X, intNE_X, intSW_X, intSE_X], dim=0)
        positions_all_y = torch.cat([intNW_Y, intNE_Y, intSW_Y, intSE_Y], dim=0)
        weights_all = torch.cat([fltNW, fltNE, fltSW, fltSE], dim=0)
        batch_all = torch.cat(
            [batch_indices, batch_indices, batch_indices, batch_indices], dim=0
        )

        tenIn_flat_corners = torch.cat(
            [tenIn_flat, tenIn_flat, tenIn_flat, tenIn_flat], dim=0
        )

        valid_mask_all = (
            (positions_all_x >= 0)
            & (positions_all_x < W)
            & (positions_all_y >= 0)
            & (positions_all_y < H)
        )
        positions_all_x = positions_all_x[valid_mask_all]
        positions_all_y = positions_all_y[valid_mask_all]
        weights_all = weights_all[valid_mask_all]
        batch_all = batch_all[valid_mask_all]
        vals = tenIn_flat_corners[valid_mask_all] * weights_all.unsqueeze(1)

        idx_nhw = (
            batch_all.to(dtype=torch.int32) * H * W
            + positions_all_y.to(dtype=torch.int32) * W
            + positions_all_x.to(dtype=torch.int32)
        )

        tenOut_flat.index_add_(0, idx_nhw, vals)

        # Reshape tenOut back to [N, C, H, W]
        tenOut = tenOut_flat.view(N, H, W, C).permute(0, 3, 1, 2)

        return self.norm(tenOut)

if __name__ == "__main__":
    import os
    import contextlib
    @contextlib.contextmanager
    def suppress_stdout_stderr():
        """Suppress stdout and stderr by redirecting them to /dev/null."""
        with open(os.devnull, "w") as devnull:
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)
            try:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                yield
            finally:
                os.dup2(old_stdout_fd, 1)
                os.dup2(old_stderr_fd, 2)
                os.close(old_stdout_fd)
                os.close(old_stderr_fd)
    model = SoftSplat("soft").half().cuda()
    tenIn = torch.randn(1, 3, 2000, 2000).cuda().half()
    
    # Compile the model
    example_inputs = (tenIn, tenIn, tenIn)

    import time
    start_time = time.time()
    model(*example_inputs)
    print("Inference time: ", time.time() - start_time)
    """

    # Save the model
    torch.jit.save(torch.jit.trace(model,example_inputs=example_inputs), "SoftSplat.pt")
    print("Model saved successfully")
    model = torch.jit.load("SoftSplat.pt").eval()
    print("Model loaded successfully")
    start_time = time.time()
    model(*example_inputs)
    print("Inference time: ", time.time() - start_time)"""

    start_time = time.time()
    softsplat(*example_inputs, 'soft')
    softsplat(*example_inputs, 'soft')
    softsplat(*example_inputs, 'soft')
    print("Inference time: ", time.time() - start_time)
    
    
    with suppress_stdout_stderr():
        from softsplat_cupy import softsplat as softsplat_cupy
        start_time = time.time()
        softsplat_cupy(*example_inputs, 'soft')
        softsplat_cupy(*example_inputs, 'soft')
        softsplat_cupy(*example_inputs, 'soft')
    print("Inference time: ", time.time() - start_time)



