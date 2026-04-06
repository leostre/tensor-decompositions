from typing import Literal
import matplotlib.pyplot as plt

class UpdateGapScheduler:
    """
    Scheduler for determining when to update projections during training.
    
    This class manages the frequency of projection updates, which can be fixed or
    change over time according to various schedules (linear, exponential, etc.).

    The main usage - method `should_update`
    """
    
    def __init__(self, start: int, end: int, mode: Literal['fixed', 'linear', 'exponential', 'exponential2'] = "fixed",
                 batch_size=1, epochs=1, training_samples=1, verbose=False):
        """
        Initialize the update gap scheduler.
        
        Args:
            start (int): Initial update interval (iterations between updates)
            end (int): Final update interval (for non-fixed modes)
            mode (str, optional): Scheduling mode. Defaults to "fixed".
            batch_size (int, optional): Batch size used in training. Defaults to 1.
            epochs (int, optional): Number of training epochs. Defaults to 1.
            training_samples (int, optional): Number of training samples. Defaults to 1.
        """
        self.update_gap = start
        '''Initial update interval (iterations between updates)'''
        self.update_gap_end = end
        '''Final update interval (for non-fixed modes)'''
        self.mode = mode
        '''Scheduling mode ('fixed', 'linear', 'exponential', or 'exponential2')'''
        self.batch_size = batch_size
        '''batch_size (int): Batch size used in training'''
        self.epochs = epochs
        '''epochs (int): Number of training epochs'''
        self.training_samples = training_samples
        '''Number of training samples'''
        self.verbose = verbose
        
        # Compute iterations and related values
        self.iter_per_epoch = self.training_samples / self.batch_size
        '''Iterations per epoch'''
        self.total_iters = int(self.iter_per_epoch * self.epochs)
        '''Total number of iterations in training'''
        
        # Initialize the first update at iteration 0
        self.next_update = 0
        '''Iteration number for the next scheduled update'''
        
        # Only print gap end if not fixed
        if self.verbose:
            print(f"Update gap scheduler initialized with {self.update_gap} start, {self.update_gap_end} end, {self.mode} mode"
                  if self.mode != "fixed" else
                f"Update gap scheduler initialized with {self.update_gap} start"
            )
    
    def compute_gap(self, current_iter):
        """
        Compute the next update gap based on current iteration.
        
        The update gap changes over time according to the specified mode.
        
        Args:
            current_iter (int): Current iteration number
            
        Returns:
            float: The computed update gap (iterations until next update)
            
        Raises:
            ValueError: If an unknown scheduler mode is specified
        """
        if self.mode == "fixed":
            return self.update_gap
        elif self.mode == "linear":
            progress = self.next_update / self.total_iters
            return self.update_gap + (self.update_gap_end - self.update_gap) * progress
        elif self.mode == "exponential":
            progress = self.next_update / self.total_iters
            return self.update_gap * ((self.update_gap_end / self.update_gap) ** progress)
        elif self.mode == "exponential2":
            # More aggressive exponential growth by squaring the progress
            progress = (self.next_update / self.total_iters) ** 2
            return self.update_gap * ((self.update_gap_end / self.update_gap) ** progress)
        else:
            raise ValueError(f"Unknown scheduler mode: {self.mode}")
    
    def should_update(self, current_iter: int) -> bool:
        """
        Check if we should update at the current iteration.
        
        This method is called during training to determine if it's time
        to update the projections.
        """
        if current_iter >= self.next_update:
            current_gap = max(1, int(self.compute_gap(current_iter)))
            self.next_update = current_iter + current_gap
            return True
        return False
            
    def print_update_steps(self):
        """
        Simulate and print the update schedule without affecting the scheduler's state.
        
        This method is useful for debugging and visualizing the update schedule
        before training begins.
        """
        list_of_updates = []
        next_update_sim = 0
        
        # For epoch statistics
        epoch_updates = [[] for _ in range(self.epochs)]
        
        for i in range(self.total_iters):
            if i >= next_update_sim:
                progress = next_update_sim / self.total_iters
                if self.mode == "fixed":
                    current_gap = self.update_gap
                elif self.mode == "linear":
                    current_gap = self.update_gap + (self.update_gap_end - self.update_gap) * progress
                elif self.mode == "exponential":
                    current_gap = self.update_gap * ((self.update_gap_end / self.update_gap) ** progress)
                elif self.mode == "exponential2":
                    progress = progress ** 2  # Square the progress for more aggressive growth
                    current_gap = self.update_gap * ((self.update_gap_end / self.update_gap) ** progress)
                
                current_gap = max(1, int(current_gap))
                list_of_updates.append((i, current_gap))
                
                # Track updates per epoch
                current_epoch = int(i / self.iter_per_epoch)
                if current_epoch < self.epochs:
                    epoch_updates[current_epoch].append(current_gap)
                
                next_update_sim = i + current_gap
        
        print(f"Update schedule simulation: {list_of_updates}")
        
        # Print epoch statistics
        print("\nEpoch-wise statistics:")
        for epoch, gaps in enumerate(epoch_updates):
            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                print(f"Epoch {epoch}: average gap = {avg_gap:.2f} ({len(gaps)} updates)")
            
    def plot_update_schedule(self, save_path=None):
        """
        Plot the update schedule showing intervals over iterations.
        
        This method creates a visualization of how the update interval
        changes over the course of training.
        
        Args:
            save_path (str, optional): If provided, saves the plot to this path.
                                     If None, displays the plot.
        """
        # Simulate the schedule
        iterations = []
        gaps = []
        next_update_sim = 0
        
        for i in range(self.total_iters):
            if i >= next_update_sim:
                progress = next_update_sim / self.total_iters
                if self.mode == "fixed":
                    current_gap = self.update_gap
                elif self.mode == "linear":
                    current_gap = self.update_gap + (self.update_gap_end - self.update_gap) * progress
                elif self.mode == "exponential":
                    current_gap = self.update_gap * ((self.update_gap_end / self.update_gap) ** progress)
                elif self.mode == "exponential2":
                    progress = progress ** 2
                    current_gap = self.update_gap * ((self.update_gap_end / self.update_gap) ** progress)
                
                current_gap = max(1, int(current_gap))
                iterations.append(i)
                gaps.append(current_gap)
                next_update_sim = i + current_gap

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, gaps, 'b.-', label='Update Interval')
        
        plt.title(f'Update Interval Schedule ({self.mode} mode)')
        plt.xlabel('Iteration')
        plt.ylabel('Update Interval')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Either save or display the plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()