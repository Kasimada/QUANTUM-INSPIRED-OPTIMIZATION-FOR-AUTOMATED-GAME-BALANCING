from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple


class Optimizer(ABC):
    """Base class for all optimization algorithms."""

    @abstractmethod
    def optimize(
        self, objective_fn: Callable, initial_state: Any, config: Dict[str, Any]
    ) -> Tuple[Any, float, list, list]:
        """
        Runs the optimization process.

        Args:
            objective_fn: Function to evaluate fitness
            initial_state: The starting state/data for the optimizer
            config: Dictionary containing optimizer settings

        Returns:
            Tuple containing (best_state, best_score, history_x, history_y) or similar format
        """
        pass
