import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger

# for debug nan issues
from ..debug_tools.nan_debugger import NaNDebugger
# from torch_musa.utils.compare_tool import CompareWithCPU, open_module_tracker, ModuleInfo, NanInfTracker

DEBUG_FLASH_ATTN = os.environ.get("MUSA_FLASH_ATTENTION_DEBUG", "0") == "1"
if DEBUG_FLASH_ATTN:
    print("FlashAttention debug mode enabled")

def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    
    # for accuracy alignment, we set sampler by manual and set shuffle=False, 
    # and ensure the order of data is the same across epochs. 
    # This is important for some special training process.
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers) # Set shuffle=False to ensure the order of data
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    # for debug
    if DEBUG_FLASH_ATTN:
        debugger = NaNDebugger(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            model_logger=model_logger
        )
        debugger.enable_hooks()
        debugger.enable_grad_hooks()
        debugger.enable_backward_hooks() # add bwd hook
    
    for epoch_id in range(num_epochs):
        # for data in tqdm(dataloader):
        for step, data in enumerate(tqdm(dataloader)):
            
            if DEBUG_FLASH_ATTN:
                debugger.save_shadow_state(step, epoch_id, data)
            
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                
                if DEBUG_FLASH_ATTN:
                    debugger.check_loss(loss)    
                
                accelerator.backward(loss)
                
                if DEBUG_FLASH_ATTN:
                    if debugger.nan_detected:
                        if accelerator.is_main_process:
                            debugger.dump_nan_state()

                        accelerator.wait_for_everyone()

                        raise RuntimeError("NaN detected, debug state saved") 
                
                optimizer.step()
                accelerator.print(
                    f" epoch={epoch_id} step={step} loss={loss.item():.6f}"
                )
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
