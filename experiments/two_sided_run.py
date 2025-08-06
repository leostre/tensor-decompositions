import sys
import os

# Правильный путь с учетом вложенности
correct_path = "/home/user/projects/tensor-decompositions/tensor-decompositions"
sys.path.insert(0, correct_path)

# Проверка
print("Обновленный sys.path:")
print(sys.path[0])

from tdecomp.matrix.decomposer import TwoSidedRandomSVD
import sys
import os
from typing import List, Dict, Tuple, Callable, Any
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
def setup_logging():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/experiment_{timestamp}.txt"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()
torch.manual_seed(42)
RANDOM_INIT_METHODS = ['normal', 'ortho', 'iid_entries', 'lean_walsh', 'identity_copies']
N_REPEATS = 20
SQUARE_SIZES = [2**i for i in range(4, 13)]
RECTANGULAR_FIXED_DIM = 256
RECTANGULAR_VARIABLE_SIZES = [2**i for i in range(4, 13)]

class ExperimentRunner:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True 
        
    def clear_cuda_cache(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def warmup(self):
        if self.device == "cuda":
            x = torch.randn(100, 100, device=self.device)
            for _ in range(10):
                _ = torch.linalg.svd(x)
            self.clear_cuda_cache()
        
    def run_method(
        self,
        method: Callable,
        matrix_generator: Callable,
        method_kwargs: Dict[str, Any] = None,
        n_repeats: int = 1
    ) -> List[Dict]:
        method_kwargs = method_kwargs or {}
        results = []
        
        for _ in range(n_repeats):
            self.clear_cuda_cache()
            X = matrix_generator()
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            if method == torch.linalg.svd:
                U, S, Vh = method(X, full_matrices=False)
                X_approx = U @ torch.diag(S) @ Vh
            else:
                decomposer = method(**method_kwargs)
                U, S, Vh = decomposer.decompose(X)
                X_approx = decomposer._two_sided_compose(U, S, Vh)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            
            error = torch.norm(X - X_approx).item()
            rel_error = error / torch.norm(X).item()
            rank = len(S)
            
            results.append({
                "shape": tuple(X.shape),
                "error": error,
                "relative_error": rel_error,
                "time": elapsed,
                "rank": rank
            })
        
        return results
    
    def aggregate_results(self, results: List[Dict]) -> Dict:
        return {
            "shape": results[0]["shape"],
            "error_mean": np.mean([r["error"] for r in results]),
            "error_std": np.std([r["error"] for r in results]),
            "time_mean": np.mean([r["time"] for r in results]),
            "time_std": np.std([r["time"] for r in results]),
            "rank_mean": np.mean([r["rank"] for r in results]),
            "rank_std": np.std([r["rank"] for r in results]),
            "raw_results": results
        }
    
    def generate_matrix(self, shape: Tuple[int, int]) -> Callable:
        def generator():
            return torch.randn(*shape, device=self.device, dtype=torch.float32)
        return generator
    
    def run_comparative_experiment(self):
        self.warmup()
        all_results = {method: [] for method in RANDOM_INIT_METHODS}
        baseline_results = []
        
        for i, size in enumerate(SQUARE_SIZES):
            logger.info(f"Processing square matrices {i+1}/{len(SQUARE_SIZES)} size {size}x{size}")
            matrix_gen = self.generate_matrix((size, size))
            
            logger.info("Running baseline SVD...")
            baseline = self.run_method(
                torch.linalg.svd,
                matrix_gen,
                n_repeats=N_REPEATS
            )
            baseline_results.append(self.aggregate_results(baseline))
            
            for method in RANDOM_INIT_METHODS:
                logger.info(f"Running {method} initialization...")
                results = self.run_method(
                    TwoSidedRandomSVD,
                    matrix_gen,
                    method_kwargs={"distortion_factor": 0.6, "random_init": method},
                    n_repeats=N_REPEATS
                )
                all_results[method].append(self.aggregate_results(results))
                self.clear_cuda_cache()
        
        for i, size in enumerate(RECTANGULAR_VARIABLE_SIZES):
            logger.info(f"Processing rectangular matrices {i+1}/{len(RECTANGULAR_VARIABLE_SIZES)} size {RECTANGULAR_FIXED_DIM}x{size}")
            matrix_gen = self.generate_matrix((RECTANGULAR_FIXED_DIM, size))
            
            for method in RANDOM_INIT_METHODS:
                logger.info(f"Running {method} initialization...")
                results = self.run_method(
                    TwoSidedRandomSVD,
                    matrix_gen,
                    method_kwargs={"distortion_factor": 0.6, "random_init": method},
                    n_repeats=N_REPEATS
                )
                all_results[method].append(self.aggregate_results(results))
                self.clear_cuda_cache()
        
        self.plot_results(all_results, baseline_results)
        self.save_results(all_results, baseline_results)
        return all_results, baseline_results
    
    def plot_results(self, method_results: Dict, baseline_results: List[Dict]):
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        for method, data in method_results.items():
            square_data = [d for d in data[:len(SQUARE_SIZES)]]
            sizes = [d["shape"][0] for d in square_data]
            errors = [d["error_mean"] for d in square_data]
            errors_std = [d["error_std"] for d in square_data]
            plt.errorbar(sizes, errors, yerr=errors_std, fmt='o-', label=method)
        
        sizes = [d["shape"][0] for d in baseline_results]
        errors = [d["error_mean"] for d in baseline_results]
        errors_std = [d["error_std"] for d in baseline_results]
        plt.errorbar(sizes, errors, yerr=errors_std, fmt='k--', label='SVD Baseline')
        
        plt.xlabel('Matrix Size (N×N)')
        plt.ylabel('Error')
        plt.title('Comparison of Errors (Square Matrices)')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        
        plt.subplot(2, 2, 2)
        for method, data in method_results.items():
            square_data = [d for d in data[:len(SQUARE_SIZES)]]
            sizes = [d["shape"][0] for d in square_data]
            times = [d["time_mean"] for d in square_data]
            times_std = [d["time_std"] for d in square_data]
            plt.errorbar(sizes, times, yerr=times_std, fmt='o-', label=method)
        
        sizes = [d["shape"][0] for d in baseline_results]
        times = [d["time_mean"] for d in baseline_results]
        times_std = [d["time_std"] for d in baseline_results]
        plt.errorbar(sizes, times, yerr=times_std, fmt='k--', label='SVD Baseline')
        
        plt.xlabel('Matrix Size (N×N)')
        plt.ylabel('Time (s)')
        plt.title('Comparison of Runtime (Square Matrices)')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        
        plt.subplot(2, 2, 3)
        for method, data in method_results.items():
            rect_data = [d for d in data[len(SQUARE_SIZES):]]
            sizes = [f"{d['shape'][0]}×{d['shape'][1]}" for d in rect_data]
            errors = [d["error_mean"] for d in rect_data]
            errors_std = [d["error_std"] for d in rect_data]
            plt.errorbar(range(len(sizes)), errors, yerr=errors_std, fmt='o-', label=method)
        
        plt.xticks(range(len(sizes)), sizes, rotation=45)
        plt.xlabel('Matrix Size (M×N)')
        plt.ylabel('Error')
        plt.title('Comparison of Errors (Rectangular Matrices)')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig(f"visualizations/comparison_results_{timestamp}.png")
        plt.close()
    
    def save_results(self, method_results: Dict, baseline_results: List[Dict]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
            
        csv_filename = f"results/results_summary_{timestamp}.csv"
            
        rows = []
            
        for res in baseline_results:
            rows.append({
                    "matrix_shape": f"{res['shape'][0]}x{res['shape'][1]}",
                    "method": "svd_baseline",
                    "error_mean": res["error_mean"],
                    "error_std": res["error_std"],
                    "time_mean": res["time_mean"],
                    "time_std": res["time_std"],
                    "rank_mean": res["rank_mean"],
                    "rank_std": res["rank_std"],
                    "matrix_type": "square" if res["shape"][0] == res["shape"][1] else "rectangular",
                    "dim1": res["shape"][0],
                    "dim2": res["shape"][1]
                })
            
        for method, data in method_results.items():
            for res in data:
                rows.append({
                        "matrix_shape": f"{res['shape'][0]}x{res['shape'][1]}",
                        "method": method,
                        "error_mean": res["error_mean"],
                        "error_std": res["error_std"],
                        "time_mean": res["time_mean"],
                        "time_std": res["time_std"],
                        "rank_mean": res["rank_mean"],
                        "rank_std": res["rank_std"],
                        "matrix_type": "square" if res["shape"][0] == res["shape"][1] else "rectangular",
                        "dim1": res["shape"][0],
                        "dim2": res["shape"][1]
                    })
            
        import csv
        with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = [
                    'matrix_shape', 'method', 'error_mean', 'error_std',
                    'time_mean', 'time_std', 'rank_mean', 'rank_std',
                    'matrix_type', 'dim1', 'dim2'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                writer.writerows(rows)
            
                txt_filename = f"results/results_summary_{timestamp}.txt"

        with open(txt_filename, "w") as f:
            f.write("=== EXPERIMENT RESULTS SUMMARY ===\n\n")
            f.write(f"Number of repeats: {N_REPEATS}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"CSV results saved to: {csv_filename}\n\n")
                
            matrix_shapes = set(row['matrix_shape'] for row in rows)
                
            for shape in sorted(matrix_shapes, key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1]))):
                shape_rows = [row for row in rows if row['matrix_shape'] == shape]
                f.write(f"\n=== MATRIX SHAPE: {shape} ===\n")
                    
                sorted_rows = sorted(shape_rows, key=lambda x: (x['method'] != 'svd_baseline', x['method']))
                   
                for row in sorted_rows:
                    f.write(
                            f"{row['method']:<15} | "
                            f"Error: {row['error_mean']:.3e} ± {row['error_std']:.1e} | "
                            f"Time: {row['time_mean']:.3f} ± {row['time_std']:.3f}s | "
                            f"Rank: {row['rank_mean']:.1f} ± {row['rank_std']:.1f}\n"
                        )
            
        logger.info(f"Results saved to {csv_filename} (CSV) and {txt_filename} (text)")    
runner = ExperimentRunner()
runner.run_comparative_experiment()