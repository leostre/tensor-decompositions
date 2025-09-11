import sys
import os
import sys
import os
import csv
import itertools
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Callable, Any, Optional


# Правильный путь с учетом вложенности
correct_path = "/home/user/projects/tensor-decompositions/tensor-decompositions"
sys.path.insert(0, correct_path)

# Проверка
print("Обновленный sys.path:")
print(sys.path[0])
from tdecomp.matrix.decomposer import TwoSidedRandomSVD

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

RANDOM_INIT_METHODS = ['normal', 'ortho', 'iid_entries', 'identity_copies']
N_REPEATS = 20
SQUARE_SIZES = [2**i for i in range(4, 11)]
RECTANGULAR_FIXED_DIM = 256
RECTANGULAR_VARIABLE_SIZES = [2**i for i in range(4, 11)]

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
            del x  # Освобождаем память после разогрева
            self.clear_cuda_cache()
        
    def run_method(
        self,
        method: Callable,
        matrix_generator: Callable,
        method_kwargs: Dict[str, Any] = None,
        n_repeats: int = 1,
        max_attempts: int = 3  
    ) -> Optional[List[Dict]]:
        method_kwargs = method_kwargs or {}
        results = []
        
        for attempt in range(max_attempts):
            try:
                for _ in range(n_repeats):
                    self.clear_cuda_cache()
                    try:
                        # Генерация матрицы
                        X = matrix_generator()
                        
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            allocated = torch.cuda.memory_allocated() / 1024**3
                            logger.debug(f"Attempt {attempt+1}: GPU memory - Total: {total_mem:.1f}GB, Used: {allocated:.1f}GB")
                        
                        # Проверка доступной памяти перед запуском SVD
                        if method == torch.linalg.svd and self.device == "cuda":
                            m, n = X.shape
                            required_mem = (m*n + m*m + n*n + min(m,n)) * 4 / 1024**3
                            free_mem = total_mem - allocated
                            
                            if required_mem > free_mem * 0.9:
                                logger.warning(f"Not enough memory for SVD. Required: {required_mem:.2f}GB, Available: {free_mem:.2f}GB")
                                del X
                                self.clear_cuda_cache()
                                raise MemoryError("Not enough memory for SVD")
                        
                        if self.device == "cuda":
                            torch.cuda.synchronize()
                        start_time = time.perf_counter()
                        
                        if method == torch.linalg.svd:
                            try:
                                with torch.cuda.amp.autocast(enabled=False):  
                                    U, S, Vh = method(X, full_matrices=False)
                                X_approx = U @ torch.diag(S) @ Vh
                                
                                del U, Vh
                            except RuntimeError as e:
                                if 'out of memory' in str(e).lower():
                                    logger.warning(f"OOM in SVD for {X.shape}, attempt {attempt+1}")
                                    if 'X_approx' in locals(): 
                                        del X_approx
                                        self.clear_cuda_cache()
                                    if 'S' in locals(): 
                                        del S
                                    del X
                                    self.clear_cuda_cache()
                                    raise MemoryError(f"OOM in SVD: {str(e)}") from e
                                raise  # Пробрасываем оригинальную ошибку, если это не OOM
                        else:
                            try:
                                decomposer = method(**method_kwargs)
                                U, S, Vh = decomposer.decompose(X)
                                X_approx = decomposer._two_sided_compose(U, S, Vh)
                                
                                del decomposer, U, Vh
                            except RuntimeError as e:
                                if 'out of memory' in str(e).lower():
                                    logger.warning(f"OOM in {method.__name__} for {X.shape}, attempt {attempt+1}")
                                    if 'X_approx' in locals(): 
                                        del X_approx
                                        self.clear_cuda_cache()
                                    if 'S' in locals(): 
                                        del S
                                    if 'decomposer' in locals(): 
                                        del decomposer
                                    del X
                                    self.clear_cuda_cache()
                                    raise MemoryError(f"OOM in {method.__name__}: {str(e)}") from e
                                raise  # Пробрасываем оригинальную ошибку
                        
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
                        
                        del X, X_approx, S
                        self.clear_cuda_cache()
                    
                    except MemoryError as e:
                        self.clear_cuda_cache()
                        logger.error(f"Memory error in run_method: {str(e)}")
                        raise  # Пробрасываем MemoryError дальше
                    
                return results  
                
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                self.clear_cuda_cache()
                if attempt == max_attempts - 1:
                    logger.error(f"All {max_attempts} attempts failed for method {method.__name__}")
                    raise  # Пробрасываем исключение после всех попыток
                continue
        
        return None
    
    def aggregate_results(self, results: List[Dict]) -> Dict:
        aggregated = {
            "shape": results[0]["shape"],
            "error_mean": np.mean([r["error"] for r in results]),
            "error_std": np.std([r["error"] for r in results]),
            "time_mean": np.mean([r["time"] for r in results]),
            "time_std": np.std([r["time"] for r in results]),
            "rank_mean": np.mean([r["rank"] for r in results]),
            "rank_std": np.std([r["rank"] for r in results]),
            "raw_results": results
        }
        del results  # Освобождаем память после агрегации
        return aggregated
    
    def generate_matrix(self, shape: Tuple[int, int], rank: Optional[int] = None) -> Callable:
        """Генератор матриц с возможностью задания ранга"""
        def generator():
            if rank is None or rank >= min(shape):  
                matrix = torch.randn(*shape, device=self.device, dtype=torch.float32)
            else:  
                A = torch.randn(shape[0], rank, device=self.device, dtype=torch.float32)
                B = torch.randn(rank, shape[1], device=self.device, dtype=torch.float32)
                matrix = A @ B
                del A, B  # Освобождаем промежуточные матрицы
            return matrix
        return generator
    
    def run_comparative_experiment(self):
        self.warmup()
        all_results = {}
        baseline_results = []
        
        def adjust_rank(method: str, rank: Optional[int], size: int) -> Optional[int]:
            if rank is None:
                return None
                
            if method == 'lean_walsh':
                if rank <= 1:
                    return 1
                adjusted_rank = 1 << (rank - 1).bit_length()
                
                if adjusted_rank > rank:
                    adjusted_rank = adjusted_rank >> 1
                return min(max(adjusted_rank, 1), size)
            
            return rank
        
        rank_strategies = [
            ('full_rank', None), 
            ('half_rank', lambda s: max(1, s // 2)),  
            ('fixed_rank', lambda s: 32) 
        ]
        
        def process_experiment(size, is_rectangular=False):
            for method in RANDOM_INIT_METHODS:
                for rank_name, rank_fn in rank_strategies:
                    shape = (RECTANGULAR_FIXED_DIM, size) if is_rectangular else (size, size)
                    current_rank = rank_fn(min(shape)) if rank_fn else None
                    adjusted_rank = adjust_rank(method, current_rank, min(shape))
                    
                    if method == 'lean_walsh' and adjusted_rank is not None:
                        if not (adjusted_rank > 0 and (adjusted_rank & (adjusted_rank - 1) == 0)):
                            logger.error(f"Invalid rank for lean_walsh: {adjusted_rank}")
                            continue
                    
                    key = f"{'rect_' if is_rectangular else ''}{method}_{rank_name}"
                    logger.info(f"Processing {key} {shape[0]}x{shape[1]} (rank={adjusted_rank})")
                    
                    try:
                        matrix_gen = self.generate_matrix(shape, rank=adjusted_rank)
                        
                        if method == RANDOM_INIT_METHODS[0]:
                            logger.info(f"Running baseline SVD for {shape[0]}x{shape[1]} ({rank_name})")
                            baseline = self.run_method(
                                torch.linalg.svd,
                                matrix_gen,
                                n_repeats=N_REPEATS
                            )
                            if baseline is not None:
                                baseline_results.append({
                                    **self.aggregate_results(baseline),
                                    "rank_strategy": rank_name,
                                    "method": "svd_baseline",
                                    "matrix_shape": f"{shape[0]}x{shape[1]}"
                                })
                                del baseline  
                        
                        results = self.run_method(
                            TwoSidedRandomSVD,
                            matrix_gen,
                            method_kwargs={
                                "random_init": method,
                                "rank": adjusted_rank  
                            },
                            n_repeats=N_REPEATS
                        )
                        
                        if results is not None:
                            if key not in all_results:
                                all_results[key] = []
                            
                            aggregated = self.aggregate_results(results)
                            all_results[key].append({
                                **aggregated,
                                "rank_strategy": rank_name,
                                "method": method,
                                "matrix_shape": f"{shape[0]}x{shape[1]}",
                                "actual_rank": adjusted_rank if adjusted_rank is not None else min(shape)
                            })
                            del results, aggregated 
                    
                    except Exception as e:
                        logger.error(f"Error processing {shape[0]}x{shape[1]} with {method}: {str(e)}")
                        self.clear_cuda_cache()
                        continue
                    
                    self.clear_cuda_cache()
        
        try:
            for size in SQUARE_SIZES:
                process_experiment(size)
            
            for size in RECTANGULAR_VARIABLE_SIZES:
                process_experiment(size, is_rectangular=True)
            
            self.plot_results(all_results, baseline_results)
            self.save_results(all_results, baseline_results)
            return all_results, baseline_results
        
        except Exception as e:
            logger.critical(f"Critical error in run_comparative_experiment: {str(e)}")
            self.clear_cuda_cache()
            raise 
            
        finally:
            self.clear_cuda_cache()
            torch.cuda.empty_cache()
    
    def plot_results(self, method_results: Dict, baseline_results: List[Dict]):
        plt.figure(figsize=(18, 12))
        
        for i, rank_name in enumerate(['full_rank', 'half_rank', 'fixed_rank']):
            plt.subplot(2, 3, i+1)
            plt.title(f'Square matrices ({rank_name})')
            
            base_sizes = [int(d["matrix_shape"].split('x')[0]) 
                        for d in baseline_results 
                        if d["rank_strategy"] == rank_name and 'x' in d["matrix_shape"] and 
                        d["matrix_shape"].split('x')[0] == d["matrix_shape"].split('x')[1]]
            base_errors = [d["error_mean"] 
                        for d in baseline_results 
                        if d["rank_strategy"] == rank_name and 'x' in d["matrix_shape"] and 
                        d["matrix_shape"].split('x')[0] == d["matrix_shape"].split('x')[1]]
            plt.plot(base_sizes, base_errors, 'k--', label='SVD Baseline')
            
            for method in RANDOM_INIT_METHODS:
                key = f"{method}_{rank_name}"
                if key in method_results:
                    data = method_results[key]
                    sizes = [int(d["matrix_shape"].split('x')[0]) for d in data]
                    errors = [d["error_mean"] for d in data]
                    plt.plot(sizes, errors, 'o-', label=method)
            
            plt.xlabel('Matrix Size')
            plt.ylabel('Error')
            plt.legend()
            plt.grid(True)
            plt.xscale('log')
            plt.yscale('log')
        
        for i, rank_name in enumerate(['full_rank', 'half_rank', 'fixed_rank']):
            plt.subplot(2, 3, i+4)
            plt.title(f'Rectangular matrices ({rank_name})')
            
            base_sizes = [d["matrix_shape"] 
                        for d in baseline_results 
                        if d["rank_strategy"] == rank_name and 'x' in d["matrix_shape"] and 
                        d["matrix_shape"].split('x')[0] != d["matrix_shape"].split('x')[1]]
            base_errors = [d["error_mean"] 
                        for d in baseline_results 
                        if d["rank_strategy"] == rank_name and 'x' in d["matrix_shape"] and 
                        d["matrix_shape"].split('x')[0] != d["matrix_shape"].split('x')[1]]
            plt.plot(range(len(base_sizes)), base_errors, 'k--', label='SVD Baseline')
            plt.xticks(range(len(base_sizes)), base_sizes, rotation=45)
            
            for method in RANDOM_INIT_METHODS:
                key = f"rect_{method}_{rank_name}"
                if key in method_results:
                    data = method_results[key]
                    sizes = [d["matrix_shape"] for d in data]
                    errors = [d["error_mean"] for d in data]
                    plt.plot(range(len(sizes)), errors, 'o-', label=method)
                    plt.xticks(range(len(sizes)), sizes, rotation=45)
            
            plt.xlabel('Matrix Size')
            plt.ylabel('Error')
            plt.legend()
            plt.grid(True)
            plt.yscale('log')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("visualizations", exist_ok=True)
        error_plot_path = f"visualizations/comparison_results_{timestamp}.png"
        plt.savefig(error_plot_path)
        plt.close()
        logger.info(f"Saved error visualization to {error_plot_path}")
        
        plt.figure(figsize=(18, 12))
        
        for i, rank_name in enumerate(['full_rank', 'half_rank', 'fixed_rank']):
            plt.subplot(2, 3, i+1)
            plt.title(f'Square matrices ({rank_name}) - Time')
            
            base_sizes = [int(d["matrix_shape"].split('x')[0]) 
                        for d in baseline_results 
                        if d["rank_strategy"] == rank_name and 'x' in d["matrix_shape"] and 
                        d["matrix_shape"].split('x')[0] == d["matrix_shape"].split('x')[1]]
            base_times = [d["time_mean"] 
                        for d in baseline_results 
                        if d["rank_strategy"] == rank_name and 'x' in d["matrix_shape"] and 
                        d["matrix_shape"].split('x')[0] == d["matrix_shape"].split('x')[1]]
            plt.plot(base_sizes, base_times, 'k--', label='SVD Baseline')
            
            for method in RANDOM_INIT_METHODS:
                key = f"{method}_{rank_name}"
                if key in method_results:
                    data = method_results[key]
                    sizes = [int(d["matrix_shape"].split('x')[0]) for d in data]
                    times = [d["time_mean"] for d in data]
                    plt.plot(sizes, times, 'o-', label=method)
            
            plt.xlabel('Matrix Size')
            plt.ylabel('Time (s)')
            plt.legend()
            plt.grid(True)
            plt.xscale('log')
            plt.yscale('log')
        
        for i, rank_name in enumerate(['full_rank', 'half_rank', 'fixed_rank']):
            plt.subplot(2, 3, i+4)
            plt.title(f'Rectangular matrices ({rank_name}) - Time')
            
            base_sizes = [d["matrix_shape"] 
                        for d in baseline_results 
                        if d["rank_strategy"] == rank_name and 'x' in d["matrix_shape"] and 
                        d["matrix_shape"].split('x')[0] != d["matrix_shape"].split('x')[1]]
            base_times = [d["time_mean"] 
                        for d in baseline_results 
                        if d["rank_strategy"] == rank_name and 'x' in d["matrix_shape"] and 
                        d["matrix_shape"].split('x')[0] != d["matrix_shape"].split('x')[1]]
            plt.plot(range(len(base_sizes)), base_times, 'k--', label='SVD Baseline')
            plt.xticks(range(len(base_sizes)), base_sizes, rotation=45)
            
            for method in RANDOM_INIT_METHODS:
                key = f"rect_{method}_{rank_name}"
                if key in method_results:
                    data = method_results[key]
                    sizes = [d["matrix_shape"] for d in data]
                    times = [d["time_mean"] for d in data]
                    plt.plot(range(len(sizes)), times, 'o-', label=method)
                    plt.xticks(range(len(sizes)), sizes, rotation=45)
            
            plt.xlabel('Matrix Size')
            plt.ylabel('Time (s)')
            plt.legend()
            plt.grid(True)
            plt.yscale('log')
        
        plt.tight_layout()
        time_plot_path = f"visualizations/time_comparison_{timestamp}.png"
        plt.savefig(time_plot_path)
        plt.close()
        logger.info(f"Saved time visualization to {time_plot_path}")

    def save_results(self, method_results: Dict, baseline_results: List[Dict]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
        
        rows = []
        
        for res in baseline_results:
            rows.append({
                "matrix_shape": res["matrix_shape"],
                "method": res["method"],
                "rank_strategy": res["rank_strategy"],
                "error_mean": res["error_mean"],
                "error_std": res["error_std"],
                "time_mean": res["time_mean"],
                "time_std": res["time_std"],
                "rank_mean": res["rank_mean"],
                "rank_std": res["rank_std"],
                "matrix_type": "square" if res["matrix_shape"].split('x')[0] == res["matrix_shape"].split('x')[1] else "rectangular"
            })
        
        for key, data_list in method_results.items():
            for res in data_list:
                rows.append({
                    "matrix_shape": res["matrix_shape"],
                    "method": res["method"],
                    "rank_strategy": res["rank_strategy"],
                    "error_mean": res["error_mean"],
                    "error_std": res["error_std"],
                    "time_mean": res["time_mean"],
                    "time_std": res["time_std"],
                    "rank_mean": res["rank_mean"],
                    "rank_std": res["rank_std"],
                    "matrix_type": "square" if res["matrix_shape"].split('x')[0] == res["matrix_shape"].split('x')[1] else "rectangular"
                })
        
        csv_filename = f"results/results_summary_{timestamp}.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = [
                'matrix_shape', 'method', 'rank_strategy', 
                'error_mean', 'error_std', 'time_mean', 'time_std',
                'rank_mean', 'rank_std', 'matrix_type'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        txt_filename = f"results/results_summary_{timestamp}.txt"
        with open(txt_filename, "w") as f:
            f.write("=== EXPERIMENT RESULTS SUMMARY ===\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Number of repeats: {N_REPEATS}\n")
            f.write(f"Device: {self.device}\n\n")
            
            for rank_name in ['full_rank', 'half_rank', 'fixed_rank']:
                f.write(f"\n=== RANK STRATEGY: {rank_name} ===\n")
                
                all_shapes = sorted(set(r['matrix_shape'] for r in rows if r['rank_strategy'] == rank_name),
                                  key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1])))
                
                for shape in all_shapes:
                    f.write(f"\n  Matrix {shape}:\n")
                    shape_rows = [r for r in rows if r['matrix_shape'] == shape and r['rank_strategy'] == rank_name]
                    
                    for row in sorted(shape_rows, key=lambda x: (x['method'] != 'svd_baseline', x['method'])):
                        f.write(f"    {row['method']:<20} | "
                               f"Error: {row['error_mean']:.3e} ± {row['error_std']:.1e} | "
                               f"Time: {row['time_mean']:.3f}s ± {row['time_std']:.3f} | "
                               f"Rank: {row['rank_mean']:.1f}\n")
        
        logger.info(f"Results saved to:\nCSV: {csv_filename}\nText: {txt_filename}")

if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_comparative_experiment()