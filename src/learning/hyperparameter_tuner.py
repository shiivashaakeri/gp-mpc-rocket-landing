"""
GP Hyperparameter Retraining for Online Learning

Periodically re-optimizes GP hyperparameters as new data
is collected. This ensures the GP model remains well-calibrated
over time.

Methods:
1. Maximum Likelihood Estimation (MLE)
2. Maximum A Posteriori (MAP) with priors
3. Cross-validation based tuning

Reference:
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes
    for Machine Learning. MIT Press.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter tuning."""

    # Optimization method
    method: str = "mle"  # "mle", "map", "cv"

    # Optimization settings
    max_iter: int = 100
    learning_rate: float = 0.1
    tol: float = 1e-4

    # Bounds for hyperparameters
    lengthscale_bounds: Tuple[float, float] = (0.01, 10.0)
    variance_bounds: Tuple[float, float] = (0.001, 10.0)
    noise_bounds: Tuple[float, float] = (1e-6, 1.0)

    # Retraining triggers
    min_data_for_retrain: int = 100
    retrain_interval: int = 500  # Points between retraining

    # Cross-validation
    cv_folds: int = 5


class HyperparameterTuner:
    """
    Tunes GP hyperparameters using various methods.

    Supports:
    - Maximum Likelihood Estimation (MLE)
    - Maximum A Posteriori (MAP)
    - Cross-validation

    Example:
        >>> tuner = HyperparameterTuner(config)
        >>>
        >>> # Tune hyperparameters
        >>> new_params = tuner.tune(gp_model, X_train, Y_train)
        >>> gp_model.set_hyperparameters(new_params)
    """

    def __init__(self, config: Optional[HyperparameterConfig] = None):
        """
        Initialize hyperparameter tuner.

        Args:
            config: Configuration parameters
        """
        self.config = config or HyperparameterConfig()

        # Tracking
        self._tuning_history: List[Dict] = []
        self._points_since_tune = 0

    def should_retrain(self, n_new_points: int) -> bool:
        """
        Check if retraining should be triggered.

        Args:
            n_new_points: Number of new data points

        Returns:
            True if retraining should occur
        """
        self._points_since_tune += n_new_points

        return self._points_since_tune >= self.config.retrain_interval

    def tune(
        self,
        gp_model,
        X: NDArray,
        Y: NDArray,
    ) -> Dict[str, Any]:
        """
        Tune GP hyperparameters.

        Args:
            gp_model: GP model to tune
            X: Training inputs
            Y: Training outputs

        Returns:
            Optimized hyperparameters
        """
        if len(X) < self.config.min_data_for_retrain:
            return self._get_current_hyperparams(gp_model)

        start_time = time.time()

        if self.config.method == "mle":
            new_params = self._tune_mle(gp_model, X, Y)
        elif self.config.method == "map":
            new_params = self._tune_map(gp_model, X, Y)
        elif self.config.method == "cv":
            new_params = self._tune_cv(gp_model, X, Y)
        else:
            new_params = self._get_current_hyperparams(gp_model)

        tune_time = time.time() - start_time

        # Record history
        self._tuning_history.append(
            {
                "params": new_params,
                "n_data": len(X),
                "time": tune_time,
                "method": self.config.method,
            }
        )

        self._points_since_tune = 0

        return new_params

    def _get_current_hyperparams(self, gp_model) -> Dict[str, Any]:
        """Extract current hyperparameters from GP model."""
        params = {}

        if hasattr(gp_model, "kernel"):
            kernel = gp_model.kernel
            if hasattr(kernel, "lengthscales"):
                params["lengthscales"] = kernel.lengthscales.copy()
            if hasattr(kernel, "variance"):
                params["variance"] = kernel.variance
            if hasattr(kernel, "lengthscale"):
                params["lengthscale"] = kernel.lengthscale

        if hasattr(gp_model, "noise_variance"):
            params["noise_variance"] = gp_model.noise_variance

        return params

    def _tune_mle(
        self,
        gp_model,
        X: NDArray,
        Y: NDArray,
    ) -> Dict[str, Any]:
        """
        Tune via Maximum Likelihood Estimation.

        Maximizes: log p(Y | X, θ)
        """
        # Get initial parameters
        params = self._get_current_hyperparams(gp_model)

        # Convert to optimization vector
        theta, param_names = self._params_to_vector(params)

        # Gradient descent
        for iteration in range(self.config.max_iter):
            # Compute negative log likelihood and gradient
            nll, grad = self._compute_nll_gradient(gp_model, X, Y, theta, param_names)

            # Update with gradient descent
            theta_new = theta - self.config.learning_rate * grad

            # Apply bounds
            theta_new = self._apply_bounds(theta_new, param_names)

            # Check convergence
            if np.linalg.norm(theta_new - theta) < self.config.tol:
                break

            theta = theta_new

        # Convert back to parameters
        return self._vector_to_params(theta, param_names)

    def _tune_map(
        self,
        gp_model,
        X: NDArray,
        Y: NDArray,
    ) -> Dict[str, Any]:
        """
        Tune via Maximum A Posteriori.

        Maximizes: log p(Y | X, θ) + log p(θ)
        """
        params = self._get_current_hyperparams(gp_model)
        theta, param_names = self._params_to_vector(params)

        for iteration in range(self.config.max_iter):
            # NLL gradient
            nll, grad = self._compute_nll_gradient(gp_model, X, Y, theta, param_names)

            # Add prior gradient (log-normal prior)
            prior_grad = self._compute_prior_gradient(theta, param_names)
            grad = grad + prior_grad

            theta_new = theta - self.config.learning_rate * grad
            theta_new = self._apply_bounds(theta_new, param_names)

            if np.linalg.norm(theta_new - theta) < self.config.tol:
                break

            theta = theta_new

        return self._vector_to_params(theta, param_names)

    def _tune_cv(
        self,
        gp_model,
        X: NDArray,
        Y: NDArray,
    ) -> Dict[str, Any]:
        """
        Tune via Cross-Validation.

        Minimizes cross-validation prediction error.
        """
        n_folds = self.config.cv_folds

        params = self._get_current_hyperparams(gp_model)
        theta, param_names = self._params_to_vector(params)

        best_theta = theta.copy()
        best_cv_error = np.inf

        # Grid search + refinement
        for _ in range(self.config.max_iter // 10):
            # Random perturbation
            theta_candidate = theta + np.random.randn(len(theta)) * 0.1
            theta_candidate = self._apply_bounds(theta_candidate, param_names)

            # Compute CV error
            cv_error = self._compute_cv_error(gp_model, X, Y, theta_candidate, param_names, n_folds)

            if cv_error < best_cv_error:
                best_cv_error = cv_error
                best_theta = theta_candidate.copy()

        return self._vector_to_params(best_theta, param_names)

    def _params_to_vector(
        self,
        params: Dict[str, Any],
    ) -> Tuple[NDArray, List[str]]:
        """Convert parameter dict to optimization vector."""
        theta = []
        names = []

        if "lengthscales" in params:
            ls = np.atleast_1d(params["lengthscales"])
            for i, l in enumerate(ls):
                theta.append(np.log(l))  # Log transform for positivity
                names.append(f"log_lengthscale_{i}")
        elif "lengthscale" in params:
            theta.append(np.log(params["lengthscale"]))
            names.append("log_lengthscale")

        if "variance" in params:
            theta.append(np.log(params["variance"]))
            names.append("log_variance")

        if "noise_variance" in params:
            theta.append(np.log(params["noise_variance"]))
            names.append("log_noise")

        return np.array(theta), names

    def _vector_to_params(
        self,
        theta: NDArray,
        names: List[str],
    ) -> Dict[str, Any]:
        """Convert optimization vector back to parameters."""
        params = {}
        lengthscales = []

        for i, name in enumerate(names):
            value = np.exp(theta[i])  # Inverse log transform

            if "lengthscale" in name:
                if "_" in name and name.split("_")[-1].isdigit():
                    lengthscales.append(value)
                else:
                    params["lengthscale"] = value
            elif "variance" in name and "noise" not in name:
                params["variance"] = value
            elif "noise" in name:
                params["noise_variance"] = value

        if lengthscales:
            params["lengthscales"] = np.array(lengthscales)

        return params

    def _apply_bounds(self, theta: NDArray, names: List[str]) -> NDArray:
        """Apply bounds to hyperparameters."""
        theta_bounded = theta.copy()

        for i, name in enumerate(names):
            if "lengthscale" in name:
                bounds = self.config.lengthscale_bounds
            elif "variance" in name and "noise" not in name:
                bounds = self.config.variance_bounds
            elif "noise" in name:
                bounds = self.config.noise_bounds
            else:
                continue

            # Bounds in log space
            theta_bounded[i] = np.clip(theta[i], np.log(bounds[0]), np.log(bounds[1]))

        return theta_bounded

    def _compute_nll_gradient(
        self,
        gp_model,
        X: NDArray,
        Y: NDArray,
        theta: NDArray,
        names: List[str],
    ) -> Tuple[float, NDArray]:
        """Compute negative log likelihood and gradient."""
        # Numerical gradient
        eps = 1e-5
        grad = np.zeros_like(theta)

        nll_0 = self._compute_nll(gp_model, X, Y, theta, names)

        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            nll_plus = self._compute_nll(gp_model, X, Y, theta_plus, names)
            grad[i] = (nll_plus - nll_0) / eps

        return nll_0, grad

    def _compute_nll(
        self,
        gp_model,
        X: NDArray,
        Y: NDArray,
        theta: NDArray,
        names: List[str],
    ) -> float:
        """Compute negative log likelihood."""
        # Create temporary GP with given hyperparameters
        params = self._vector_to_params(theta, names)

        try:
            # Simple NLL approximation using prediction error
            gp_copy = self._create_gp_copy(gp_model, params)
            gp_copy.fit(X, Y)

            # Leave-one-out approximation
            mean, var = gp_copy.predict(X)

            # NLL = 0.5 * sum((y - mean)^2 / var + log(var))
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            if mean.ndim == 1:
                mean = mean.reshape(-1, 1)
            if var.ndim == 1:
                var = var.reshape(-1, 1)

            diff = Y - mean
            nll = 0.5 * np.sum(diff**2 / (var + 1e-10) + np.log(var + 1e-10))

            return nll

        except Exception:
            return 1e10  # Large penalty for failure

    def _compute_prior_gradient(
        self,
        theta: NDArray,
        names: List[str],  # noqa: ARG002
    ) -> NDArray:
        """Compute gradient of log prior (log-normal)."""
        # Log-normal prior: penalize deviation from mean
        prior_mean = 0.0  # In log space, corresponds to value 1.0
        prior_std = 1.0

        grad = (theta - prior_mean) / (prior_std**2)

        return grad

    def _compute_cv_error(
        self,
        gp_model,
        X: NDArray,
        Y: NDArray,
        theta: NDArray,
        names: List[str],
        n_folds: int,
    ) -> float:
        """Compute cross-validation error."""
        n = len(X)
        fold_size = n // n_folds

        params = self._vector_to_params(theta, names)
        cv_errors = []

        for fold in range(n_folds):
            # Split data
            val_start = fold * fold_size
            val_end = min((fold + 1) * fold_size, n)

            val_idx = np.arange(val_start, val_end)
            train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n)])

            X_train, Y_train = X[train_idx], Y[train_idx]
            X_val, Y_val = X[val_idx], Y[val_idx]

            try:
                # Train and predict
                gp_copy = self._create_gp_copy(gp_model, params)
                gp_copy.fit(X_train, Y_train)

                mean, _ = gp_copy.predict(X_val)

                # MSE
                error = np.mean((Y_val - mean) ** 2)
                cv_errors.append(error)

            except Exception:
                cv_errors.append(1e10)

        return np.mean(cv_errors)

    def _create_gp_copy(self, gp_model, params: Dict) -> Any:
        """Create a copy of GP with new hyperparameters."""
        # This is a simplified version - full implementation would
        # properly copy and update the GP model

        try:
            import copy  # noqa: PLC0415

            gp_copy = copy.deepcopy(gp_model)

            # Update kernel parameters
            if hasattr(gp_copy, "kernel"):
                if "lengthscales" in params:
                    gp_copy.kernel.lengthscales = params["lengthscales"]
                if "lengthscale" in params:
                    gp_copy.kernel.lengthscale = params["lengthscale"]
                if "variance" in params:
                    gp_copy.kernel.variance = params["variance"]

            if "noise_variance" in params:
                gp_copy.noise_variance = params["noise_variance"]

            return gp_copy

        except Exception:
            return gp_model

    def get_tuning_history(self) -> List[Dict]:
        """Get hyperparameter tuning history."""
        return self._tuning_history


class AdaptiveHyperparameterScheduler:
    """
    Schedules hyperparameter updates based on model performance.

    Triggers retuning when:
    - Prediction error increases significantly
    - Data distribution shifts
    - Periodic intervals
    """

    def __init__(
        self,
        tuner: HyperparameterTuner,
        error_threshold: float = 0.5,
        check_interval: int = 100,
    ):
        """
        Initialize scheduler.

        Args:
            tuner: Hyperparameter tuner
            error_threshold: Error increase to trigger retuning
            check_interval: Steps between checks
        """
        self.tuner = tuner
        self.error_threshold = error_threshold
        self.check_interval = check_interval

        self._baseline_error: Optional[float] = None
        self._steps_since_check = 0

    def check_and_tune(
        self,
        gp_model,
        X: NDArray,
        Y: NDArray,
        recent_errors: NDArray,
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if tuning needed and perform if so.

        Args:
            gp_model: GP model
            X: Training data
            Y: Training targets
            recent_errors: Recent prediction errors

        Returns:
            (did_tune, new_params)
        """
        self._steps_since_check += 1

        if self._steps_since_check < self.check_interval:
            return False, None

        self._steps_since_check = 0

        # Compute current error
        current_error = np.mean(recent_errors**2)

        # Initialize baseline
        if self._baseline_error is None:
            self._baseline_error = current_error
            return False, None

        # Check for error increase
        error_increase = current_error / (self._baseline_error + 1e-10)

        if error_increase > 1 + self.error_threshold:
            # Error increased significantly - retune
            new_params = self.tuner.tune(gp_model, X, Y)
            self._baseline_error = current_error
            return True, new_params

        # Periodic retuning
        if self.tuner.should_retrain(0):
            new_params = self.tuner.tune(gp_model, X, Y)
            self._baseline_error = current_error
            return True, new_params

        return False, None
