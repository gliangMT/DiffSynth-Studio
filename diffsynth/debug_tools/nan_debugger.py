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
        target_keywords = None,
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
        self.target_keywords = target_keywords  # 可配置的关键词列表
    
    def _should_hook(self, name):
        if self.target_keywords is None:
            return True
        
        name = name.lower()
        for kw in self.target_keywords:
            if kw in name:
                return True
        return False
    
    # ------------------------------------------------
    # 获取真实模型（兼容 Accelerate / DeepSpeed）
    # ------------------------------------------------
    def _get_raw_model(self):

        if self.accelerator is not None:
            try:
                return self.accelerator.unwrap_model(self.model)
            except Exception:
                return self.model

        return self.model
    
    # ------------------------------------------------
    # 保存 shadow checkpoint（NaN-1）
    # ------------------------------------------------
    def save_shadow_state(self, step, epoch, batch):

        model = self._get_raw_model()

        if self.accelerator is not None:
            model_state = self.accelerator.get_state_dict(self.model)
        else:
            model_state = model.state_dict()

        self.last_good_state = {
            "model": copy.deepcopy(model_state),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "step": step,
            "epoch": epoch,
        }

        self.last_good_batch = copy.deepcopy(batch)

        self.nan_detected = False  # 每次 shadow 更新时重置

    # ------------------------------------------------
    # 保存 NaN 现场
    # ------------------------------------------------
    def dump_nan_state(self):

        if self.last_good_state is None:
            print("[Nan Debugger] WARNING: last_good_state is None, skip dump")
            return

        path_ckpt = os.path.join(self.save_dir, "nan_minus_1.pt")
        path_batch = os.path.join(self.save_dir, "nan_batch.pt")

        print("\n===== NaN detected! Saving debug state =====")

        # 只在主进程保存
        if self.accelerator is None or self.accelerator.is_main_process:

            # 保存 shadow 数据
            torch.save(self.last_good_state, path_ckpt)
            torch.save(self.last_good_batch, path_batch)

            print("Saved ckpt:", path_ckpt)
            print("Saved batch:", path_batch)

            # 获取真实模型（ZeRO2 下可用, zero3则需要额外处理）
            model = self._get_raw_model()

            # 保存当前参数（用于恢复）
            current_state = model.state_dict()

            # 临时加载 NaN-1 权重
            with torch.no_grad():
                model.load_state_dict(self.last_good_state["model"], strict=True)

            # 保存 NaN-1 模型
            self.model_logger.save_model(
                self.accelerator,
                model,
                f"nan_detected_step-{self.last_good_state['step']}.safetensors"
            )

            # 恢复当前模型
            with torch.no_grad():
                model.load_state_dict(current_state, strict=True)

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
                if not torch.isfinite(t).all():

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
        
        count = 0
        for name, module in self.model.named_modules():
            if not self._should_hook(name):
                continue
            module.register_forward_hook(self._forward_hook(name))
            count += 1

        print(f"[MUSA DEBUG] Forward hooks enabled for {count} modules")
        self.hooks_enabled = True

    # ------------------------------------------------
    # grad hook
    # ------------------------------------------------
    def enable_grad_hooks(self):

        print("Enabling gradient NaN hooks")
        count = 0
        for name, p in self.model.named_parameters():
            if not self._should_hook(name):
                continue
            
            if p.requires_grad:
                p.register_hook(self._grad_hook(name))
                count += 1
        print(f"[MUSA DEBUG] Gradient hooks enabled for {count} parameters")
    
    def _grad_hook(self, name):

        def hook(grad):
            if grad is None:
                return

            if torch.isnan(grad).any():
                print("\n===== NaN detected in gradient =====")
                print("param:", name)
                print("shape:", grad.shape)
                self.nan_detected = True
                return

        return hook
    
    # ------------------------------------------------
    # backward hook
    # ------------------------------------------------
    def enable_backward_hooks(self):

        print("Enabling backward hooks")
        count = 0
        for name, module in self.model.named_modules():
            if not self._should_hook(name):
                continue
            module.register_full_backward_hook(self._bwd_hook(name))
            count += 1
        print(f"[MUSA DEBUG] Backward hooks enabled for {count} modules")
    
    def _bwd_hook(self, name):

        def hook(module, grad_input, grad_output):

            tensors = []

            if isinstance(grad_output, tuple):
                tensors += [t for t in grad_output if isinstance(t, torch.Tensor)]
            elif isinstance(grad_output, torch.Tensor):
                tensors.append(grad_output)

            for g in tensors:

                if g is not None and torch.isnan(g).any():

                    print("\n===== NaN detected in backward =====")
                    print("module:", name)
                    print("grad shape:", g.shape)

                    try:
                        print("min:", g.nanmin().item(),
                            "max:", g.nanmax().item())
                    except:
                        pass

                    self.nan_detected = True  # 只标记
                    return

        return hook