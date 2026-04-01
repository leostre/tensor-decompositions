import pandas as pd
import matplotlib.pyplot as plt

class MetricVisualizer:

    @staticmethod
    def plot_metric_by_run(
        df: pd.DataFrame,
        target: str = "loss",
        step_col: str = "step",
        run_id_col: str = "run_id",
        legend_col: str = "run_name",
        figsize: tuple[int, int] = (12, 7),
    ):
        required = {run_id_col, step_col, target}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"В DataFrame нет колонок: {sorted(missing)}")

        plot_df = df[[run_id_col, step_col, target, legend_col]].copy()
        plot_df[run_id_col] = plot_df[run_id_col].astype(str)
        # plot_df = plot_df.dropna(subset=[run_id_col, step_col, target])
        plot_df[step_col] = pd.to_numeric(plot_df[step_col], errors="coerce")
        plot_df[target] = pd.to_numeric(plot_df[target], errors="coerce")
        # plot_df = plot_df.dropna(subset=[step_col, target])
        plot_df = plot_df.sort_values(step_col)

        fig, ax = plt.subplots(figsize=figsize)
        for run_id, g in plot_df.groupby(run_id_col, sort=False):
            ax.plot(g[step_col], g[target], label=g[legend_col][0], linewidth=1.5)

        ax.set_xlabel(step_col)
        ax.set_ylabel(target)
        ax.set_title(f"{target} vs {step_col} for each {legend_col}")
        ax.grid(True, alpha=0.3)
        ax.legend(title=legend_col, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        plt.show()

        return fig, ax