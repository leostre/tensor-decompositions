import itertools
import math
import mlflow
from transformers import PreTrainedModel, TrainerCallback
import psutil
from torch import nn
import torch
import torch.nn.functional as F
import pandas as pd

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


class PerplexityCallback(TrainerCallback):
    def __init__(
        self,
        dataloader=None,
        max_batches=2,
        eval_every_n_steps=5,
        log_key="perplexity",
    ):
        self.dataloader = dataloader
        self.max_batches = max_batches
        self.eval_every_n_steps = eval_every_n_steps
        self.log_key = log_key

    def on_step_end(self, args, state, control, model=None, eval_dataloader=None, train_dataloader=None, **kwargs):
        if state.global_step % self.eval_every_n_steps != 0:
            return

        if model is None:
            return

        dataloader = self.dataloader or eval_dataloader or train_dataloader
        if dataloader is None:
            return

        was_training = model.training
        model.eval()

        total_nll = 0.0
        total_tokens = 0
        #COPY PARTIALLY FROM LabelSmoother class in transformers lib
        with torch.no_grad():
            for batch in itertools.islice(dataloader, self.max_batches):
                batch = {k: v for k, v in batch.items()}

                outputs = model(**batch)
                logits = outputs.logits
                labels = batch["labels"]

                #SHIFT
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
                shift_mask = batch['attention_mask'][..., 1:].contiguous().bool()


                log_probs = -nn.functional.log_softmax(logits, dim=-1)
                labels = torch.clamp(labels, min=0) #for safe indexing in torch.gather
                #print("labels shape", labels.shape, "logprobs shape", log_probs.shape)
                nll_loss = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # (1, L_gen-1)
                
                masked_logprobs = nll_loss.masked_select(shift_mask)
                total_nll += masked_logprobs.sum() #total per batch, not per dim=-1
                total_tokens += shift_mask.sum()

        if was_training:
            model.train()

        ppl = torch.exp(total_nll / total_tokens).item()
        if mlflow.active_run() is not None:
            mlflow.log_metrics({self.log_key: ppl}, step=int(state.global_step))


class PreciseMemoryCallback(TrainerCallback):
    def __init__(self, prefix="mem"):
        self.prefix = prefix
        self.g = None

    @staticmethod
    def _nbytes(t):
        return t.numel() * t.element_size()

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(next(model.parameters()).device)
        

    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        if state.global_step != 2:
            #measure only once, because values are const (except activations)
            return

        w = sum(self._nbytes(p) for p in model.parameters())
        g = sum(self._nbytes(p.grad) for p in model.parameters() if p.grad is not None)

        opt_total, opt_cuda = 0, 0
        for st in optimizer.state.values():
            # there will be M, V of Adam for every p in parameters() and it x2 because of first_proj, second_proj
            for v in st.values():
                if torch.is_tensor(v):
                    b = self._nbytes(v)
                    opt_total += b
                    if v.is_cuda:
                        opt_cuda += b

        metrics = {
            f"{self.prefix}_weights_mb": w / 1024**2,
            f"{self.prefix}_grads_mb": g / 1024**2,
            f"{self.prefix}_opt_state_mb": opt_total / 1024**2,
        }

        if torch.cuda.is_available():
            dev = next(model.parameters()).device
            alloc = torch.cuda.memory_allocated(dev)
            peak = torch.cuda.max_memory_allocated(dev)
            activ_est = max(0, alloc - (w + g + opt_cuda))
            activ_peak_est = max(0, peak - (w + g + opt_cuda))
            metrics.update({
                f"{self.prefix}_activations_estimate_mb": activ_est / 1024**2,
                f"{self.prefix}_activations_peak_during_one_step_est_mb": activ_peak_est / 1024**2,
            })

        mlflow.log_params(metrics)


def print_param_shapes(optimizer, only_group_index=None):
    if (only_group_index is not None):
        [print(tens.shape) for tens in optimizer.param_groups[only_group_index]['params']]
        print(len(optimizer.param_groups[only_group_index]['params']))
        return
    for i, param_group in enumerate(optimizer.param_groups):
        print("group", i)
        [print(tens.shape) for tens in param_group['params']]
        print(len(param_group['params']))

def get_batch_size_mb(batch_encoding: dict):
    """Получить размер BatchEncoding в МБ (суммирует все тензоры)
    Args:
        batch_encoding: состоит из input_ids, attention_mask, labels
    """
    total_bytes = sum(
        v.numel() * v.element_size() 
        for v in batch_encoding.values() 
        if torch.is_tensor(v)
    )
    return total_bytes / (1024 ** 2)

def get_model_size_mb(model):
    """Подсчитать примерный размер весов модели в МБ"""
    total_params = 0
    total_bytes = 0
    
    for param in model.parameters():
        num_params = param.numel()
        param_bytes = num_params * param.element_size()
        total_params += num_params
        total_bytes += param_bytes
    
    size_mb = total_bytes / (1024 ** 2)
    
    return size_mb