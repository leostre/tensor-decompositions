import mlflow
from transformers import PreTrainedModel, TrainerCallback
import psutil
import torch

from tdecomp.grad_proj.tensorgrad.projectors.update_gap_scheduler import UpdateGapScheduler
from tdecomp.grad_proj.tensorgrad.tensorgrad import TensorGRaD

class UpdateGapMLflowCallback(TrainerCallback):
    def __init__(self):
        self._prev_next_update = None

    def _get_scheduler(self, optimizer: TensorGRaD) -> UpdateGapScheduler:
        #group in optimizer 0 contains some strange consts
        #group 1 is good
        param_0_tensor = optimizer.param_groups[1]["params"][0]
        return getattr(optimizer.state[param_0_tensor]['first_proj'], "update_gap_scheduler", None)

    def on_train_begin(self, args, state, control, optimizer=None, **kwargs):
        self._prev_next_update = 0
        # scheduler: UpdateGapScheduler = self._get_scheduler(optimizer)
        # if scheduler is None:
        #     print("~~~~~UpdateGapScheduler is None on_train_begin callback~~~~~~~~")
        #     return 0
        # if scheduler is not None:
        #     self._prev_next_update = int(scheduler.next_update)

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        scheduler = self._get_scheduler(optimizer)

        cur_next_update = int(scheduler.next_update)
        changed = int(
            self._prev_next_update is not None and cur_next_update != self._prev_next_update
        )

        if mlflow.active_run() is not None:
            mlflow.log_metrics(
                {
                    "ug_next_update": cur_next_update,
                    "ug_next_update_changed": changed,
                },
                step=int(state.global_step),
            )

        self._prev_next_update = cur_next_update


class SystemMetricsCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if mlflow.active_run() is not None:            
            metrics = {
                "gpu_memory_used_mb": torch.cuda.memory_allocated() / 1e6,
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1e6,
                "cpu_percent": psutil.cpu_percent(),
            }
            mlflow.log_metrics(metrics, step=int(state.global_step))