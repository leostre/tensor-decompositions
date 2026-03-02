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


class MlflowMetricsPandasConverter:
    def __init__(self, mlFlowClient: mlflow.MlflowClient):
        self.mlFlowClient = mlFlowClient

    def get_metrics_from_run(self, run_id: str, main_metric="loss", include_parameters=[]) -> pd.DataFrame:
        '''main_metric - metric that defines amount of steps in result dataframe'''
        run_example = self.mlFlowClient.get_run(run_id)
        metric_names = list(run_example.data.metrics.keys())
        df = pd.DataFrame({"step": [m.step for m in self.mlFlowClient.get_metric_history(run_id, main_metric)]})
        # print(df)
        for metric_name in metric_names:
            metric_log = self.mlFlowClient.get_metric_history(run_id, metric_name)
            metric_df = pd.DataFrame([{"step": m.step, metric_name: m.value} for m in metric_log])
            df = df.merge(metric_df, on="step", how="left")

        if (len(include_parameters) > 0):
            df_stats = pd.DataFrame([self.mlFlowClient.get_run(run_id).data.params] * len(df))
            df_stats = df_stats[include_parameters]
            df = pd.concat([df, df_stats], axis=1)
        
        return df

    def get_metrics_from_all_experiment_runs(self, experiment_id, filter_string='', include_parameters=[]) -> pd.DataFrame:
        '''It suppoused, that all exepiments with equal metric columns'''
        runs = self.mlFlowClient.search_runs(experiment_id, filter_string=filter_string)
        combined_df = pd.DataFrame()
        for run in runs:
            df = self.get_metrics_from_run(run.info.run_id, include_parameters = include_parameters)
            df["run_id"] = run.info.run_id

            combined_df = pd.concat([combined_df, df])
        return combined_df
    
    def get_last_metrics_from_all_experiment_runs(self, experiment_id, filter_string='') -> pd.DataFrame:
        runs = self.mlFlowClient.search_runs(experiment_id, filter_string=filter_string)
        combined_df = pd.DataFrame()
        for run in runs:
            df = pd.DataFrame([self.mlFlowClient.get_run(run.info.run_id).data.metrics])
            df.insert(0, "run_id", [run.info.run_id])
            combined_df = pd.concat([combined_df, df])
        return combined_df