import os
import copy
import torch
from accelerate import Accelerator


class NaNDebugger:

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer=None,
        scheduler=None,
        accelerator: Accelerator = None,
        save_dir="./nan_debug",
        model_logger=None,
    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.model_logger = model_logger

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.last_good_state = None
        self.last_good_batch = None

        self.nan_detected = False
        self.nan_location = None

        self.forward_hooks_enabled = False
        self.grad_hooks_enabled = False

    # ------------------------------------------------
    # 保存 NaN-1 shadow state
    # ------------------------------------------------

    def save_shadow_state(self, step, epoch, batch):

        if self.accelerator is not None:
            model_state = self.accelerator.get_state_dict(self.model)
        else:
            model_state = self.model.state_dict()

        self.last_good_state = {
            "model": model_state,
            "optimizer": self.optimizer.state_dict()
            if self.optimizer
            else None,
            "scheduler": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "step": step,
            "epoch": epoch,
        }

        self.last_good_batch = copy.deepcopy(batch)

    # ------------------------------------------------
    # dump NaN debug 信息
    # ------------------------------------------------

    def dump_nan_state(self):

        if self.last_good_state is None:
            print("[NaNDebugger] last_good_state is None")
            return

        if self.accelerator and not self.accelerator.is_main_process:
            return

        print("\n========== NaN detected ==========")
        print("location:", self.nan_location)

        ckpt_path = os.path.join(self.save_dir, "nan_minus_1.pt")
        batch_path = os.path.join(self.save_dir, "nan_batch.pt")

        torch.save(self.last_good_state, ckpt_path)
        torch.save(self.last_good_batch, batch_path)

        print("Saved ckpt:", ckpt_path)
        print("Saved batch:", batch_path)

        if self.model_logger is not None:

            model = (
                self.accelerator.unwrap_model(self.model)
                if self.accelerator
                else self.model
            )

            original_state = model.state_dict()

            model.load_state_dict(self.last_good_state["model"])

            self.model_logger.save_model(
                self.accelerator,
                self.model,
                f"nan_step_{self.last_good_state['step']}.safetensors",
            )

            model.load_state_dict(original_state)

        print("=================================\n")

    # ------------------------------------------------
    # loss check
    # ------------------------------------------------

    def check_loss(self, loss):

        if not torch.isfinite(loss):

            print("\n===== NaN/Inf loss detected =====")

            try:
                print("loss:", loss.item())
            except:
                pass

            self.nan_detected = True
            self.nan_location = "loss"

    # ------------------------------------------------
    # tensor check
    # ------------------------------------------------

    def check_tensor(self, tensor, name):

        if not isinstance(tensor, torch.Tensor):
            return

        if not torch.isfinite(tensor).all():

            print("\n===== NaN tensor detected =====")
            print("tensor:", name)
            print("shape:", tensor.shape)

            try:
                print("absmax:", tensor.abs().max().item())
            except:
                pass

            self.nan_detected = True
            self.nan_location = name

    # ------------------------------------------------
    # forward hook
    # ------------------------------------------------

    def _forward_hook(self, name):

        def hook(module, inp, out):

            if self.nan_detected:
                return

            tensors = []

            if isinstance(out, torch.Tensor):
                tensors.append(out)

            elif isinstance(out, (list, tuple)):
                tensors.extend(
                    [t for t in out if isinstance(t, torch.Tensor)]
                )

            for t in tensors:

                if not torch.isfinite(t).all():

                    print("\n===== NaN detected in forward =====")
                    print("module:", name)
                    print("shape:", t.shape)

                    try:
                        print("absmax:", t.abs().max().item())
                    except:
                        pass

                    self.nan_detected = True
                    self.nan_location = f"forward:{name}"

                    return

        return hook

    # ------------------------------------------------
    # 注册 forward hooks
    # ------------------------------------------------

    def enable_forward_hooks(self):

        if self.forward_hooks_enabled:
            return

        print("[NaNDebugger] Enabling forward hooks")

        for name, module in self.model.named_modules():

            if len(list(module.children())) == 0:
                module.register_forward_hook(self._forward_hook(name))

        self.forward_hooks_enabled = True

    # ------------------------------------------------
    # gradient hooks
    # ------------------------------------------------

    def _grad_hook(self, name):

        def hook(grad):

            if grad is None or self.nan_detected:
                return

            if not torch.isfinite(grad).all():

                print("\n===== NaN detected in gradient =====")
                print("param:", name)
                print("shape:", grad.shape)

                try:
                    print("absmax:", grad.abs().max().item())
                except:
                    pass

                self.nan_detected = True
                self.nan_location = f"grad:{name}"

        return hook

    # ------------------------------------------------
    # 注册 gradient hooks
    # ------------------------------------------------

    def enable_grad_hooks(self):

        if self.grad_hooks_enabled:
            return

        print("[NaNDebugger] Enabling gradient hooks")

        for name, p in self.model.named_parameters():

            if p.requires_grad:
                p.register_hook(self._grad_hook(name))

        self.grad_hooks_enabled = True

    # ------------------------------------------------
    # step 结束检查
    # ------------------------------------------------

    def step_check(self):

        if self.nan_detected:

            self.dump_nan_state()

            raise RuntimeError(
                f"NaN detected at {self.nan_location}"
            )