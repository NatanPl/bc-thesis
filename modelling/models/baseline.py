import numpy as np
import torch

try:
    from statsmodels.tsa.api import VAR
except ImportError:
    VAR = None


def _to_np(ds):
    """RawDataset -> np.ndarray (C,T,F)."""
    arr = ds.data
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=float)


# ------------------------------------------------------------------ #
# Naïve persistence baseline                                         #
# ------------------------------------------------------------------ #
def baseline_persistence(train_ds, val_ds, test_ds, horizon=1, lookback_L=5):
    """
    Forecast = last observed value at each step.
    Uses rolling walk-forward over test horizon using true history.
    For country-split datasets, use all available history from train+val.
    """
    train_arr = _to_np(train_ds)  # (C_train, T_train, F)
    val_arr = _to_np(val_ds)      # (C_val, T_val, F)
    test_arr = _to_np(test_ds)    # (C_test, T_test, F)

    # Check if we can concatenate along time axis (same countries)
    if train_arr.shape[0] == val_arr.shape[0] == test_arr.shape[0]:
        # Traditional time-based split
        hist = np.concatenate([train_arr, val_arr], axis=1)  # (C,Thist,F)
        if lookback_L is not None:
            hist = hist[:, -lookback_L:, :]
        last = hist[:, -1, :]
    else:
        # Country-based split: use all available data as history
        # Concatenate all train and val data across countries and time
        all_hist = np.concatenate([
            train_arr.reshape(-1, train_arr.shape[-1]),  # (C_train*T_train, F)
            val_arr.reshape(-1, val_arr.shape[-1])       # (C_val*T_val, F)
        ], axis=0)
        if lookback_L is not None:
            all_hist = all_hist[-lookback_L:, :]
        # Use the most recent values as starting point (mean of recent observations)
        # Use last 100 observations or all if fewer
        last = np.mean(all_hist[-100:], axis=0)
        last = np.broadcast_to(last, (test_arr.shape[0], test_arr.shape[2]))

    C, Ttest, F = test_arr.shape
    preds = np.empty_like(test_arr)

    for t in range(Ttest):
        preds[:, t, :] = last
        # update with ground truth (true walk-forward)
        last = test_arr[:, t, :]

    return preds  # (C,Ttest,F)


# ------------------------------------------------------------------ #
# AR(1) per-feature pooled baseline                                  #
# ------------------------------------------------------------------ #
def baseline_ar1(train_ds, val_ds, test_ds, horizon=1, lookback_L=5):
    """
    Fit x_t = a + phi * x_{t-1} per feature pooled across all countries.
    Walk-forward multi-step; horizon=1 assumed in Experiment C.
    Handles both time-split and country-split datasets.
    """
    train_arr = _to_np(train_ds)  # (C_train, T_train, F)
    val_arr = _to_np(val_ds)      # (C_val, T_val, F)
    test_arr = _to_np(test_ds)    # (C_test, T_test, F)

    # Check if we can concatenate along time axis (same countries)
    if train_arr.shape[0] == val_arr.shape[0] == test_arr.shape[0]:
        # Traditional time-based split
        hist = np.concatenate([train_arr, val_arr], axis=1)  # (C,T,F)
        if lookback_L is not None:
            hist = hist[:, -lookback_L:, :]
        C, T, F = hist.shape
        # design
        Xlag = hist[:, :-1, :].reshape(-1, F)
        Xcur = hist[:, 1:, :].reshape(-1, F)
        last = hist[:, -1, :]  # (C,F)
    else:
        # Country-based split: pool all training data
        # Flatten all training data
        # (C_train*T_train, F)
        train_flat = train_arr.reshape(-1, train_arr.shape[-1])
        # (C_val*T_val, F)
        val_flat = val_arr.reshape(-1, val_arr.shape[-1])
        all_data = np.concatenate([train_flat, val_flat], axis=0)  # (N, F)
        if lookback_L is not None:
            all_data = all_data[-lookback_L:, :]

        F = all_data.shape[1]
        # Create lagged design matrix from pooled data
        Xlag = all_data[:-1, :]  # (N-1, F)
        Xcur = all_data[1:, :]   # (N-1, F)

        # Use most recent available data as starting point for each test country
        last = np.mean(all_data[-100:], axis=0)  # (F,)
        last = np.broadcast_to(
            last, (test_arr.shape[0], test_arr.shape[2]))  # (C_test, F)

    # Fit AR(1) model
    ones = np.ones((Xlag.shape[0], 1))
    A = np.concatenate([ones, Xlag], axis=1)      # (N,2)
    coeffs, *_ = np.linalg.lstsq(A, Xcur, rcond=None)  # (2,F)
    a = coeffs[0]
    phi = coeffs[1]
    # guard against invalid coefficients
    bad = ~np.isfinite(phi)
    phi[bad] = 1.0
    a[bad] = 0.0

    # walk-forward
    preds = np.empty_like(test_arr)
    cur = last  # (C_test, F)
    for t in range(test_arr.shape[1]):
        cur = a + phi * cur
        preds[:, t, :] = cur
        # update with truth for next step
        cur = test_arr[:, t, :]

    return preds


# ------------------------------------------------------------------ #
# VAR(p) pooled baseline                                             #
# ------------------------------------------------------------------ #
def baseline_var(train_ds, val_ds, test_ds, p=2, horizon=1, lookback_L=5):
    """
    Fit a pooled VAR(p) over all (country,time) rows.
    Forecast 1-step ahead walk-forward over test segment.
    Handles both time-split and country-split datasets.
    """
    if VAR is None:
        raise ImportError("statsmodels not installed.")

    train_arr = _to_np(train_ds)  # (C_train, T_train, F)
    val_arr = _to_np(val_ds)      # (C_val, T_val, F)
    test_arr = _to_np(test_ds)    # (C_test, T_test, F)

    hist = None  # Initialize to avoid unbound variable warnings

    # Check if we can concatenate along time axis (same countries)
    if train_arr.shape[0] == val_arr.shape[0] == test_arr.shape[0]:
        # Traditional time-based split
        hist = np.concatenate([train_arr, val_arr], axis=1)  # (C,T,F)
        if lookback_L is not None:
            hist = hist[:, -lookback_L:, :]
        C, Thist, F = hist.shape
        # pooled 2D data: stack countries then time
        pooled = hist.reshape(C * Thist, F)
        is_time_split = True
    else:
        # Country-based split: pool all training data
        # (C_train*T_train, F)
        train_flat = train_arr.reshape(-1, train_arr.shape[-1])
        # (C_val*T_val, F)
        val_flat = val_arr.reshape(-1, val_arr.shape[-1])
        pooled = np.concatenate([train_flat, val_flat], axis=0)   # (N, F)
        if lookback_L is not None:
            pooled = pooled[-lookback_L:, :]
        F = pooled.shape[1]
        is_time_split = False

    model = VAR(pooled)
    p_eff = min(p, max(1, pooled.shape[0] // 5))  # crude safety
    res = model.fit(maxlags=p_eff, ic=None, trend='c')

    # walk-forward
    preds = np.empty_like(test_arr)

    if is_time_split and hist is not None:
        # Time-based split: use last p_eff observations per country
        for c in range(test_arr.shape[0]):
            country_hist = hist[c, -res.k_ar:, :]  # shape (p_eff,F)
            for t in range(test_arr.shape[1]):
                fc = res.forecast(country_hist, steps=1)  # (1,F)
                preds[c, t, :] = fc[0]
                # roll buffer with *true* obs to keep evaluation honest
                country_hist = np.vstack(
                    [country_hist[1:], test_arr[c, t, :][None, :]])
    else:
        # Country-based split: use last p_eff observations from pooled data
        # as initial buffer for all test countries
        initial_buf = pooled[-res.k_ar:, :]  # shape (p_eff, F)
        for c in range(test_arr.shape[0]):
            buf = initial_buf.copy()  # each country starts with same history
            for t in range(test_arr.shape[1]):
                fc = res.forecast(buf, steps=1)  # (1,F)
                preds[c, t, :] = fc[0]
                # roll buffer with *true* obs to keep evaluation honest
                buf = np.vstack([buf[1:], test_arr[c, t, :][None, :]])

    return preds


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def eval_baseline(preds, test_ds):
    """
    preds: (C,T,F) numpy
    test_ds: RawDataset (provides ground truth)
    Returns dict with keys matching NN metrics: test_mae, test_loss (MSE).
    """
    y_true = _to_np(test_ds)
    assert preds.shape == y_true.shape, (preds.shape, y_true.shape)
    return {
        "test_mae": _mae(y_true, preds),
        "test_loss": _mse(y_true, preds),
    }


def run_one_baseline(kind, tag, train, val, test, p=2):
    """
    kind: 'persistence' | 'ar1' | 'var'
    train/val/test: preprocessed RawDataset (no NaNs, scaled)
    """
    if kind == 'persistence':
        preds = baseline_persistence(train, val, test)
    elif kind == 'ar1':
        preds = baseline_ar1(train, val, test)
    elif kind == 'var':
        preds = baseline_var(train, val, test, p=p)
    else:
        raise ValueError(kind)

    metrics = eval_baseline(preds, test)
    print(f"[{tag}] {kind} baseline metrics: {metrics}")
    return (tag, metrics)
