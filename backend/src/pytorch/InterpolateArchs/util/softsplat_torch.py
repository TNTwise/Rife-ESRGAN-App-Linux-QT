import torch

##########################################################


objCudacache = {}



##########################################################


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

    orig_dtype = tenIn.dtype
    tenIn = tenIn.float()
    tenFlow = tenFlow.float()
    if tenMetric is not None:
        tenMetric = tenMetric.float()

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

    tenOut = softsplat_func.apply(tenIn, tenFlow)

    if mode_main in ["avg", "linear", "soft"]:
        tenNormalize = tenOut[:, -1:, :, :]

        normalize_modes = {
            None: lambda x: x + 0.0000001,
            "addeps": lambda x: x + 0.0000001,
            "zeroeps": lambda x: torch.where(
                x == 0.0, torch.tensor(1.0, device=x.device), x
            ),
            "clipeps": lambda x: x.clip(0.0000001, None),
        }

        if mode_sub in normalize_modes:
            tenNormalize = normalize_modes[mode_sub](tenNormalize)

        tenOut = tenOut[:, :-1, :, :] / tenNormalize

    return tenOut.to(orig_dtype)


class softsplat_func(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, tenIn, tenFlow):
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

        # Initialize output tensor
        tenOut = torch.zeros_like(tenIn)

        # Create meshgrid of pixel coordinates
        gridY, gridX = torch.meshgrid(
            torch.arange(H, device=device, dtype=tenIn.dtype),
            torch.arange(W, device=device, dtype=tenIn.dtype),
            indexing='ij'
        )  # [H, W]

        gridX = gridX.unsqueeze(0).unsqueeze(0).expand(N, 1, H, W)
        gridY = gridY.unsqueeze(0).unsqueeze(0).expand(N, 1, H, W)

        # Compute fltX and fltY
        fltX = gridX + tenFlow[:, 0:1, :, :]
        fltY = gridY + tenFlow[:, 1:2, :, :]

        # Flatten variables
        fltX_flat = fltX.reshape(-1)
        fltY_flat = fltY.reshape(-1)
        tenIn_flat = tenIn.permute(0, 2, 3, 1).reshape(-1, C)

        # Create batch indices
        batch_indices = torch.arange(N, device=device).view(N, 1, 1).expand(N, H, W).reshape(-1)

        # Finite mask
        finite_mask = torch.isfinite(fltX_flat) & torch.isfinite(fltY_flat)
        if not finite_mask.any():
            return tenOut

        fltX_flat = fltX_flat[finite_mask]
        fltY_flat = fltY_flat[finite_mask]
        tenIn_flat = tenIn_flat[finite_mask]
        batch_indices = batch_indices[finite_mask]

        # Compute integer positions
        intNW_X = torch.floor(fltX_flat).long()
        intNW_Y = torch.floor(fltY_flat).long()
        intNE_X = intNW_X + 1
        intNE_Y = intNW_Y
        intSW_X = intNW_X
        intSW_Y = intNW_Y + 1
        intSE_X = intNW_X + 1
        intSE_Y = intNW_Y + 1

        # Compute weights
        fltNW = (intSE_X.float() - fltX_flat) * (intSE_Y.float() - fltY_flat)
        fltNE = (fltX_flat - intSW_X.float()) * (intSW_Y.float() - fltY_flat)
        fltSW = (intNE_X.float() - fltX_flat) * (fltY_flat - intNE_Y.float())
        fltSE = (fltX_flat - intNW_X.float()) * (fltY_flat - intNW_Y.float())

        # Prepare output tensor flat
        tenOut_flat = tenOut.permute(0, 2, 3, 1).reshape(-1, C)

        # Define positions and weights
        positions = [
            (intNW_X, intNW_Y, fltNW),
            (intNE_X, intNE_Y, fltNE),
            (intSW_X, intSW_Y, fltSW),
            (intSE_X, intSE_Y, fltSE),
        ]

        H, W = int(H), int(W)

        for intX, intY, weight in positions:
            # Valid indices within image bounds
            valid_mask = (intX >= 0) & (intX < W) & (intY >= 0) & (intY < H)
            if not valid_mask.any():
                continue

            idx_b = batch_indices[valid_mask]
            idx_x = intX[valid_mask]
            idx_y = intY[valid_mask]
            w = weight[valid_mask]
            vals = tenIn_flat[valid_mask] * w.unsqueeze(1)

            # Compute linear indices
            idx_NHW = idx_b * H * W + idx_y * W + idx_x

            # Accumulate values using index_add_
            tenOut_flat.index_add_(0, idx_NHW, vals)

        # Reshape tenOut back to [N, C, H, W]
        tenOut = tenOut_flat.view(N, H, W, C).permute(0, 3, 1, 2)

        ctx.save_for_backward(tenIn, tenFlow)

        return tenOut
    # end

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(self, tenOutgrad):
        tenIn, tenFlow = self.saved_tensors

        tenOutgrad = tenOutgrad.contiguous()
        assert tenOutgrad.is_cuda == True

        tenIngrad = (
            torch.zeros_like(tenIn)
            if self.needs_input_grad[0] == True
            else None
        )
        tenFlowgrad = (
            torch.zeros_like(tenFlow)
            if self.needs_input_grad[1] == True
            else None
        )

        if tenIngrad is not None:
            N, C, H, W = tenIn.shape

            gridY, gridX = torch.meshgrid(
                torch.arange(H, device=tenIn.device),
                torch.arange(W, device=tenIn.device)
            )
            gridY = gridY.unsqueeze(0).unsqueeze(0).expand(N, 1, H, W).float()
            gridX = gridX.unsqueeze(0).unsqueeze(0).expand(N, 1, H, W).float()

            fltX = gridX + tenFlow[:, 0:1, :, :]
            fltY = gridY + tenFlow[:, 1:2, :, :]

            # Check for finite values
            finite_mask = torch.isfinite(fltX) & torch.isfinite(fltY)

            intNW_X = torch.floor(fltX).long()
            intNW_Y = torch.floor(fltY).long()
            intNE_X = intNW_X + 1
            intNE_Y = intNW_Y
            intSW_X = intNW_X
            intSW_Y = intNW_Y + 1
            intSE_X = intNW_X + 1
            intSE_Y = intNW_Y + 1

            fltNW = (intSE_X.float() - fltX) * (intSE_Y.float() - fltY)
            fltNE = (fltX - intSW_X.float()) * (intSW_Y.float() - fltY)
            fltSW = (intNE_X.float() - fltX) * (fltY - intNE_Y.float())
            fltSE = (fltX - intNW_X.float()) * (fltY - intNW_Y.float())

            # Clamp indices to valid range
            intNW_X = intNW_X.clamp(0, W - 1)
            intNW_Y = intNW_Y.clamp(0, H - 1)
            intNE_X = intNE_X.clamp(0, W - 1)
            intNE_Y = intNE_Y.clamp(0, H - 1)
            intSW_X = intSW_X.clamp(0, W - 1)
            intSW_Y = intSW_Y.clamp(0, H - 1)
            intSE_X = intSE_X.clamp(0, W - 1)
            intSE_Y = intSE_Y.clamp(0, H - 1)

            # Gather tenOutgrad at neighbor positions
            def gather(tensor, x, y):
                N, C, H, W = tensor.shape
                x = x.view(N, 1, H, W).expand(-1, C, -1, -1)
                y = y.view(N, 1, H, W).expand(-1, C, -1, -1)
                return tensor.gather(3, x).gather(2, y)

            outgrad_NW = gather(tenOutgrad, intNW_X, intNW_Y)
            outgrad_NE = gather(tenOutgrad, intNE_X, intNE_Y)
            outgrad_SW = gather(tenOutgrad, intSW_X, intSW_Y)
            outgrad_SE = gather(tenOutgrad, intSE_X, intSE_Y)

            # Compute tenIngrad
            tenIngrad = (
                outgrad_NW * fltNW + 
                outgrad_NE * fltNE + 
                outgrad_SW * fltSW + 
                outgrad_SE * fltSE
            )

            # Mask invalid values
            tenIngrad = tenIngrad * finite_mask.float()

        if tenFlowgrad is not None:
            N, C_in, H, W = tenIn.shape

            gridY, gridX = torch.meshgrid(
                torch.arange(H, device=tenIn.device),
                torch.arange(W, device=tenIn.device)
            )
            gridY = gridY.unsqueeze(0).unsqueeze(0).expand(N, 1, H, W).float()
            gridX = gridX.unsqueeze(0).unsqueeze(0).expand(N, 1, H, W).float()

            fltX = gridX + tenFlow[:, 0:1, :, :]
            fltY = gridY + tenFlow[:, 1:2, :, :]

            finite_mask = torch.isfinite(fltX) & torch.isfinite(fltY)

            intNW_X = torch.floor(fltX).long()
            intNW_Y = torch.floor(fltY).long()
            intNE_X = intNW_X + 1
            intNE_Y = intNW_Y
            intSW_X = intNW_X
            intSW_Y = intNW_Y + 1
            intSE_X = intNW_X + 1
            intSE_Y = intNW_Y + 1

            intNW_X = intNW_X.clamp(0, W - 1)
            intNW_Y = intNW_Y.clamp(0, H - 1)
            intNE_X = intNE_X.clamp(0, W - 1)
            intNE_Y = intNE_Y.clamp(0, H - 1)
            intSW_X = intSW_X.clamp(0, W - 1)
            intSW_Y = intSW_Y.clamp(0, H - 1)
            intSE_X = intSE_X.clamp(0, W - 1)
            intSE_Y = intSE_Y.clamp(0, H - 1)

            w_NW_x = -1.0 * (intSE_Y.float() - fltY)
            w_NE_x = 1.0 * (intSW_Y.float() - fltY)
            w_SW_x = -1.0 * (fltY - intNE_Y.float())
            w_SE_x = 1.0 * (fltY - intNW_Y.float())

            w_NW_y = (intSE_X.float() - fltX) * -1.0
            w_NE_y = (fltX - intSW_X.float()) * -1.0
            w_SW_y = (intNE_X.float() - fltX) * 1.0
            w_SE_y = (fltX - intNW_X.float()) * 1.0

            # Gather tenOutgrad at neighbor positions
            def gather(tensor, x, y):
                N, C, H, W = tensor.shape
                x = x.view(N, 1, H, W).expand(-1, C, -1, -1)
                y = y.view(N, 1, H, W).expand(-1, C, -1, -1)
                return tensor.gather(3, x).gather(2, y)

            tenFlowgrad = torch.zeros_like(tenFlow)

            for c in range(C_in):
                tenIn_c = tenIn[:, c:c+1, :, :]

                outgrad_NW = gather(tenOutgrad[:, c:c+1, :, :], intNW_X, intNW_Y)
                outgrad_NE = gather(tenOutgrad[:, c:c+1, :, :], intNE_X, intNE_Y)
                outgrad_SW = gather(tenOutgrad[:, c:c+1, :, :], intSW_X, intSW_Y)
                outgrad_SE = gather(tenOutgrad[:, c:c+1, :, :], intSE_X, intSE_Y)

                flowgrad_x = (
                    outgrad_NW * tenIn_c * w_NW_x + 
                    outgrad_NE * tenIn_c * w_NE_x + 
                    outgrad_SW * tenIn_c * w_SW_x + 
                    outgrad_SE * tenIn_c * w_SE_x
                )

                flowgrad_y = (
                    outgrad_NW * tenIn_c * w_NW_y + 
                    outgrad_NE * tenIn_c * w_NE_y + 
                    outgrad_SW * tenIn_c * w_SW_y + 
                    outgrad_SE * tenIn_c * w_SE_y
                )

                tenFlowgrad[:, 0:1, :, :] += flowgrad_x
                tenFlowgrad[:, 1:2, :, :] += flowgrad_y

            tenFlowgrad = tenFlowgrad * finite_mask.float()

        return tenIngrad, tenFlowgrad
        # end


    # end
