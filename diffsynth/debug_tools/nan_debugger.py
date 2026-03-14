import os
import torch
import copy
from accelerate import Accelerator
from ..diffusion.logger import ModelLogger

class NaNDebugger:
    def __init__(
        self, 
        model:torch.nn.Module, 
        optimizer, 
        scheduler=None, 
        accelerator: Accelerator = None, 
        save_dir="./models/nan_debug", 
        model_logger: ModelLogger = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.save_dir = save_dir
        self.model_logger = model_logger

        os.makedirs(save_dir, exist_ok=True)

        self.last_good_state = None
        self.last_good_batch = None

        self.hooks_enabled = False
        self.nan_detected = False
    # ------------------------------------------------
    # 保存 shadow checkpoint（NaN-1）
    # ------------------------------------------------
    def save_shadow_state(self, step, epoch, batch):
        model = self.accelerator.unwrap_model(self.model)
        if self.accelerator is not None:
            model_state = self.accelerator.get_state_dict(self.model)
        else:
            model_state = model.state_dict()

        self.last_good_state = {
            "model": model_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "step": step,
            "epoch": epoch,
        }

        self.last_good_batch = copy.deepcopy(batch)

    # ------------------------------------------------
    # 保存 NaN 现场
    # ------------------------------------------------
    def dump_nan_state(self):
        
        if self.last_good_state is None:
            print("[Nan Debugger] WARNING: last_good_state is None, skip dump")
            return
        
        if self.nan_detected:
            return

        path_ckpt = os.path.join(self.save_dir, "nan_minus_1.pt")
        path_batch = os.path.join(self.save_dir, "nan_batch.pt")

        print("\n===== NaN detected! Saving debug state =====")

        torch.save(self.last_good_state, path_ckpt)
        print("Saved ckpt:", path_ckpt)

        torch.save(self.last_good_batch, path_batch)
        print("Saved batch:", path_batch)
        
        model = self.accelerator.unwrap_model(self.model)
        original_state = model.state_dict()

        self.model.load_state_dict(self.last_good_state["model"]) # load last good state to model for debugging, step Nan-1
        
        self.model_logger.save_model(
            self.accelerator, 
            self.model, 
            f"nan_detected_step-{self.last_good_state['step']}.safetensors"
        ) # save model at NaN-1 step for debugging

        self.model.load_state_dict(original_state) # restore original state to model after saving debug info
        print("=============================================\n")

    # ------------------------------------------------
    # 检查 loss
    # ------------------------------------------------
    def check_loss(self, loss):

        if not torch.isfinite(loss):
            print("Loss is NaN or Inf")
            if self.accelerator is None or self.accelerator.is_main_process:
                self.dump_nan_state()

            raise RuntimeError("NaN loss detected")

    # ------------------------------------------------
    # 检查 tensor
    # ------------------------------------------------
    def check_tensor(self, tensor, name):

        if not isinstance(tensor, torch.Tensor):
            return

        if torch.isnan(tensor).any():

            print(f"[NaN DETECTED] tensor: {name}")
            print("shape:", tensor.shape)
            if self.accelerator is None or self.accelerator.is_main_process:
                self.dump_nan_state()

            raise RuntimeError(f"NaN tensor detected: {name}")

    # ------------------------------------------------
    # forward hook
    # ------------------------------------------------
    def _forward_hook(self, name):

        def hook(module, inp, out):

            tensors = []

            if isinstance(out, torch.Tensor):
                tensors.append(out)

            elif isinstance(out, (tuple, list)):
                tensors += [t for t in out if isinstance(t, torch.Tensor)]

            for t in tensors:
                if torch.isnan(t).any():

                    print("\n===== NaN detected in forward =====")
                    print("module:", name)
                    print("shape:", t.shape)

                    try:
                        print("min:", t.nanmin().item(), "max:", t.nanmax().item())
                    except:
                        pass
                    if self.accelerator is None or self.accelerator.is_main_process:
                        self.dump_nan_state()

                    raise RuntimeError(f"NaN detected in module {name}")

        return hook

    # ------------------------------------------------
    # 注册 hooks
    # ------------------------------------------------
    def enable_hooks(self):

        if self.hooks_enabled:
            return

        print("Enabling NaN forward hooks")

        for name, module in self.model.named_modules():

            module.register_forward_hook(self._forward_hook(name))

        self.hooks_enabled = True

    # ------------------------------------------------
    # grad hook
    # ------------------------------------------------
    def enable_grad_hooks(self):

        print("Enabling gradient NaN hooks")

        for name, p in self.model.named_parameters():

            if p.requires_grad:

                p.register_hook(self._grad_hook(name))

    def _grad_hook(self, name):

        def hook(grad):

            if grad is None:
                return

            if torch.isnan(grad).any():

                print("\n===== NaN detected in gradient =====")
                print("param:", name)
                print("shape:", grad.shape)
                if self.accelerator is None or self.accelerator.is_main_process:
                    self.dump_nan_state()

                raise RuntimeError(f"NaN gradient in {name}")

        return hook
    
    # ------------------------------------------------
    # backward hook
    # ------------------------------------------------
    def enable_backward_hooks(self):

        print("Enabling backward hooks")

        for name, module in self.model.named_modules():
            module.register_full_backward_hook(self._bwd_hook(name))
    
    def _bwd_hook(self, name):

        def hook(module, grad_input, grad_output):

            tensors = []

            if isinstance(grad_output, tuple):
                tensors += [t for t in grad_output if isinstance(t, torch.Tensor)]

            elif isinstance(grad_output, torch.Tensor):
                tensors.append(grad_output)

            for g in tensors:

                if torch.isnan(g).any():
                    print("\n===== NaN detected in backward =====")
                    print("module:", name)
                    print("grad shape:", g.shape)

                    try:
                        print("min:", g.nanmin().item(), "max:", g.nanmax().item())
                    except:
                        pass

                    # try:
                    #     self.dump_nan_state()
                    # except Exception as e:
                    #     print("Failed to dump NaN state:", e)
                    
                    # 不立即 raise
                    self.nan_detected = True

                    # raise RuntimeError(f"NaN backward in {name}")

        return hook