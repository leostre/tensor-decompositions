import pandas as pd
from transformers import PreTrainedModel, TrainerCallback
import mlflow

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
        '''It supposed, that all exepiments with equal metric columns'''
        runs = self.mlFlowClient.search_runs(experiment_id, filter_string=filter_string)
        combined_df = pd.DataFrame()
        for run in runs:
            df = self.get_metrics_from_run(run.info.run_id, include_parameters = include_parameters)
            df["run_id"] = run.info.run_id
            df['run_name'] = run.info.run_name

            combined_df = pd.concat([combined_df, df])
        return combined_df
    
    def get_only_last_metrics_from_all_experiment_runs(self, experiment_id, filter_string='') -> pd.DataFrame:
        runs = self.mlFlowClient.search_runs(experiment_id, filter_string=filter_string)
        combined_df = pd.DataFrame()
        for run in runs:
            df = pd.DataFrame([self.mlFlowClient.get_run(run.info.run_id).data.metrics])
            df.insert(0, "run_id", [run.info.run_id])
            df.insert(1, 'run_name', [run.info.run_name])
            combined_df = pd.concat([combined_df, df])
        return combined_df