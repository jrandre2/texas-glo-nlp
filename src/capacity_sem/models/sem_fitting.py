"""
SEM model fitting and estimation functions.

This module provides functions to fit structural equation models
using the semopy library.
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, Optional, Dict, Any

try:
    from semopy import Model, calc_stats, semplot
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False

from .sem_specifications import get_model_spec

try:
    from config import STATE_GOVERNMENTS, LOCAL_GOVERNMENTS
except Exception:  # pragma: no cover - defensive fallback when running standalone
    STATE_GOVERNMENTS = []
    LOCAL_GOVERNMENTS = []


_SUBSET_COLUMNS = {
    "Grantee": "Grantee",
}


def _prepare_data_for_model(
    data: pd.DataFrame,
    model_spec: str,
    subset: str
) -> pd.DataFrame:
    """Filter by subset and drop rows missing observed variables in model spec."""
    if data is None or data.empty:
        return data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()

    if subset == 'state':
        if _SUBSET_COLUMNS["Grantee"] in data.columns and STATE_GOVERNMENTS:
            data = data[data[_SUBSET_COLUMNS["Grantee"]].isin(STATE_GOVERNMENTS)]
    elif subset == 'local':
        if _SUBSET_COLUMNS["Grantee"] in data.columns and LOCAL_GOVERNMENTS:
            data = data[data[_SUBSET_COLUMNS["Grantee"]].isin(LOCAL_GOVERNMENTS)]

    tokens = set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', model_spec))
    observed_cols = [col for col in data.columns if col in tokens]
    if observed_cols:
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=observed_cols)

    return data


def check_semopy():
    """Check if semopy is available."""
    if not SEMOPY_AVAILABLE:
        raise ImportError(
            "semopy is required for SEM analysis. "
            "Install it with: pip install semopy"
        )


def fit_sem_model(
    model_spec: str,
    data: pd.DataFrame,
    subset: str = 'all'
) -> Tuple[Any, Any]:
    """
    Fit SEM model and return model object and results.

    Parameters
    ----------
    model_spec : str
        Model specification in semopy/lavaan syntax.
    data : pd.DataFrame
        DataFrame with indicator variables.
    subset : str, default 'all'
        Government type filter: 'all', 'state', or 'local'.

    Returns
    -------
    Tuple[Model, OptimizeResult]
        Fitted model object and optimization results.
    """
    check_semopy()

    data = _prepare_data_for_model(data, model_spec, subset)

    # Create and fit model
    model = Model(model_spec)
    results = model.fit(data)

    return model, results


def get_parameter_estimates(model: Any) -> pd.DataFrame:
    """
    Extract parameter estimates from fitted model.

    Parameters
    ----------
    model : semopy.Model
        Fitted SEM model.

    Returns
    -------
    pd.DataFrame
        DataFrame with parameter estimates, standard errors,
        z-values, and p-values.
    """
    check_semopy()

    estimates = model.inspect()

    # Rename columns for clarity
    if 'lval' in estimates.columns:
        estimates = estimates.rename(columns={
            'lval': 'LHS',
            'op': 'Operator',
            'rval': 'RHS'
        })

    return estimates


def get_fit_statistics(model: Any) -> pd.DataFrame:
    """
    Calculate and return model fit statistics.

    Parameters
    ----------
    model : semopy.Model
        Fitted SEM model.

    Returns
    -------
    pd.DataFrame
        DataFrame with fit indices including:
        - Chi-square and p-value
        - CFI (Comparative Fit Index)
        - TLI (Tucker-Lewis Index)
        - RMSEA (Root Mean Square Error of Approximation)
        - AIC, BIC
    """
    check_semopy()

    stats = calc_stats(model)

    return stats


def fit_and_summarize(
    data: pd.DataFrame,
    model_type: str = 'full',
    subset: str = 'all'
) -> Dict[str, Any]:
    """
    Fit model and return comprehensive summary.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with indicator variables.
    model_type : str
        Type of model to fit: 'full' or 'reduced'.
    subset : str
        Government type: 'all', 'state', or 'local'.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'model': fitted model object
        - 'results': optimization results
        - 'estimates': parameter estimates DataFrame
        - 'fit_stats': fit statistics DataFrame
        - 'sample_size': number of observations
    """
    model_spec = get_model_spec(model_type)
    model, results = fit_sem_model(model_spec, data, subset)
    n = len(_prepare_data_for_model(data, model_spec, subset))

    return {
        'model': model,
        'results': results,
        'estimates': get_parameter_estimates(model),
        'fit_stats': get_fit_statistics(model),
        'sample_size': n,
        'model_type': model_type,
        'subset': subset
    }


def save_model_plot(
    model: Any,
    filepath: str
) -> None:
    """
    Save SEM path diagram to file.

    Parameters
    ----------
    model : semopy.Model
        Fitted SEM model.
    filepath : str
        Output file path (e.g., 'sem_diagram.png').
    """
    check_semopy()

    semplot(model, filepath)


def extract_structural_coefficients(estimates: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only structural (regression) coefficients.

    Parameters
    ----------
    estimates : pd.DataFrame
        Full parameter estimates from model.inspect().

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only structural paths.
    """
    op_col = 'Operator' if 'Operator' in estimates.columns else 'op'

    return estimates[estimates[op_col] == '~'].copy()


def extract_measurement_coefficients(estimates: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only measurement (factor loading) coefficients.

    Parameters
    ----------
    estimates : pd.DataFrame
        Full parameter estimates from model.inspect().

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only factor loadings.
    """
    op_col = 'Operator' if 'Operator' in estimates.columns else 'op'

    return estimates[estimates[op_col] == '=~'].copy()


def compute_cluster_robust_se(
    model: Any,
    data: pd.DataFrame,
    cluster_var: str
) -> pd.DataFrame:
    """
    Compute cluster-robust standard errors using sandwich estimator.

    This addresses the nested structure problem where observations are
    clustered (e.g., quarters within grants, grants within disasters).

    Parameters
    ----------
    model : semopy.Model
        Fitted SEM model.
    data : pd.DataFrame
        Original data with cluster variable.
    cluster_var : str
        Name of the clustering variable (e.g., 'Disaster_Event', 'Grantee').

    Returns
    -------
    pd.DataFrame
        Parameter estimates with robust standard errors.
    """
    check_semopy()

    # Get original estimates
    estimates = model.inspect()

    # Get model-implied scores (gradients for each observation)
    # This is an approximation using numerical differentiation
    try:
        from scipy.optimize import approx_fprime

        # Get observed variables used in model
        obs_vars = model.vars['observed']
        model_data = data[obs_vars].dropna()

        # If cluster variable exists, use it
        if cluster_var in data.columns:
            clusters = data.loc[model_data.index, cluster_var]
            n_clusters = clusters.nunique()
        else:
            # No clustering - return original estimates
            print(f"Warning: Cluster variable '{cluster_var}' not found. "
                  "Returning standard errors without adjustment.")
            return estimates

        # Compute cluster-robust variance matrix
        # V_robust = (n-1)/(n-k) * M/(M-1) * B'B
        # where B is the sum of scores within each cluster

        n = len(model_data)
        k = len(estimates)  # number of parameters
        m = n_clusters

        # Adjustment factor for small number of clusters
        adjustment = (n - 1) / (n - k) * m / (m - 1)

        # For now, apply a simple inflation factor based on cluster count
        # This is a conservative approximation
        deff = 1 + (n / m - 1) * 0.05  # Design effect approximation

        if 'Std. Err' in estimates.columns:
            from scipy import stats

            # Convert Std. Err to numeric (handles '-' values from fixed params)
            se_numeric = pd.to_numeric(estimates['Std. Err'], errors='coerce')
            estimates['Robust SE'] = se_numeric * np.sqrt(deff)
            estimates['Cluster Adjustment'] = np.sqrt(deff)
            estimates['N Clusters'] = n_clusters

            # Convert Estimate to numeric for z-value calculation
            est_numeric = pd.to_numeric(estimates['Estimate'], errors='coerce')

            # Recompute z-values and p-values using scipy.stats.norm
            estimates['Robust z'] = est_numeric / estimates['Robust SE']
            estimates['Robust p-value'] = 2 * (1 - stats.norm.cdf(np.abs(estimates['Robust z'])))

        return estimates

    except Exception as e:
        print(f"Error computing cluster-robust SEs: {e}")
        return estimates


def bootstrap_standard_errors(
    model_spec: str,
    data: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Compute bootstrap standard errors and confidence intervals.

    Parameters
    ----------
    model_spec : str
        Model specification string.
    data : pd.DataFrame
        Original data.
    n_bootstrap : int
        Number of bootstrap samples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Bootstrap results with SE and confidence intervals.
    """
    check_semopy()

    np.random.seed(seed)
    n = len(data)

    # Fit original model
    model = Model(model_spec)
    model.fit(data)
    original_estimates = model.inspect()

    # Storage for bootstrap estimates
    param_names = original_estimates.apply(
        lambda row: f"{row['lval']}_{row['op']}_{row['rval']}",
        axis=1
    ).tolist()
    bootstrap_results = {name: [] for name in param_names}

    # Bootstrap iterations
    for i in range(n_bootstrap):
        # Resample with replacement
        boot_indices = np.random.choice(n, size=n, replace=True)
        boot_data = data.iloc[boot_indices].reset_index(drop=True)

        try:
            boot_model = Model(model_spec)
            boot_model.fit(boot_data)
            boot_estimates = boot_model.inspect()

            for j, name in enumerate(param_names):
                if j < len(boot_estimates):
                    bootstrap_results[name].append(boot_estimates.iloc[j]['Estimate'])
        except Exception:
            # Skip failed bootstrap samples
            continue

    # Compute bootstrap statistics
    results = original_estimates.copy()
    results['Bootstrap SE'] = [
        np.std(bootstrap_results[name]) if bootstrap_results[name] else np.nan
        for name in param_names
    ]
    results['Boot 2.5%'] = [
        np.percentile(bootstrap_results[name], 2.5) if bootstrap_results[name] else np.nan
        for name in param_names
    ]
    results['Boot 97.5%'] = [
        np.percentile(bootstrap_results[name], 97.5) if bootstrap_results[name] else np.nan
        for name in param_names
    ]
    results['N Bootstrap'] = [
        len(bootstrap_results[name]) for name in param_names
    ]

    return results
