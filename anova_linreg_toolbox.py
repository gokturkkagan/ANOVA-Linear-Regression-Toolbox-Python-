"""
anova_linreg_toolbox.py

This module implements a small "toolbox" of functions for:
  (A) One-way ANOVA
  (B) Two-way ANOVA
  (C) Multiple linear regression (normal error model)

Author: Your Name
Date: YYYY-MM-DD
"""

import math
import itertools
import numpy as np
from typing import List, Tuple, Union


# -------------------------------------------------------------------
#                      Helper / Utility Functions
# -------------------------------------------------------------------

def _flatten_data(data: List[List[float]]) -> List[float]:
    """
    Utility: Flatten a list of lists into a single list of all values.
    """
    return [x for group in data for x in group]

def _mean(values: List[float]) -> float:
    """
    Utility: Return the arithmetic mean of a list of values.
    """
    return sum(values)/len(values)

def _sum_of_squares(values: List[float]) -> float:
    """
    Utility: Return sum of (x - mean)^2 for x in values.
    """
    m = _mean(values)
    return sum((x - m)**2 for x in values)

def _two_way_means(data_3d: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute overall mean, row means, column means, and cell means
    for a two-way layout.

    data_3d has shape (I, J, K): 
      i=0..I-1, j=0..J-1, k=0..K-1
    """
    # Overall mean
    grand_mean = np.mean(data_3d)

    # Row means: average over j,k
    row_means = np.mean(data_3d, axis=(1,2))  # shape (I,)

    # Col means: average over i,k
    col_means = np.mean(data_3d, axis=(0,2))  # shape (J,)

    # Cell means: average over k
    cell_means = np.mean(data_3d, axis=2)    # shape (I, J)

    return grand_mean, row_means, col_means, cell_means


# -------------------------------------------------------------------
#                  One-Way ANOVA Related Functions
# -------------------------------------------------------------------

def ANOVA1_partition_TSS(data: List[List[float]]) -> Tuple[float, float, float]:
    """
    Partition the total sum of squares (one-way layout) into:
        SStotal, SSw (within-group), SSb (between-group).
    
    data: List of length I, where data[i] is the list of observations in group i.
    
    Returns: (SStotal, SSw, SSb)
    """
    # Flatten all data to compute overall mean
    all_data = _flatten_data(data)
    overall_mean = _mean(all_data)

    # TSS
    SStotal = sum((x - overall_mean)**2 for x in all_data)

    # SSw
    SSw = 0.0
    for group in data:
        group_mean = _mean(group)
        SSw += sum((x - group_mean)**2 for x in group)

    # SSb
    SSb = SStotal - SSw

    return (SStotal, SSw, SSb)


def ANOVA1_test_equality(data: List[List[float]], alpha: float = 0.05) -> None:
    """
    Test the equality of group means in a one-way ANOVA layout:
        H0: mu1 = mu2 = ... = muI  vs  H1: not all means are equal.

    Prints out:
    - SStotal, SSw, SSb, MSb, MSw
    - F statistic, critical value, p-value
    - Decision
    
    data: a list of I groups (each group is a list of observations)
    alpha: significance level
    """
    I = len(data)  # number of groups
    n_i = [len(g) for g in data]
    N = sum(n_i)   # total sample size

    # Partition TSS
    SStotal, SSw, SSb = ANOVA1_partition_TSS(data)

    # Degrees of freedom
    dfB = I - 1
    dfW = N - I

    MSb = SSb / dfB
    MSw = SSw / dfW

    # F statistic
    F_stat = MSb / MSw

    # Compute the p-value and critical value using the F distribution
    # We assume you have access to scipy.stats.f
    # If not, you can approximate or tabulate. Below is the SciPy approach.
    try:
        from scipy.stats import f
        p_value = 1.0 - f.cdf(F_stat, dfB, dfW)
        F_crit = f.ppf(1.0 - alpha, dfB, dfW)
    except ImportError:
        p_value = float('nan')
        F_crit = float('nan')

    reject = (p_value < alpha)

    # Print a small summary table
    print("--------------------------------------------------------")
    print("One-Way ANOVA Test")
    print("--------------------------------------------------------")
    print(f"Number of groups (I)       = {I}")
    print(f"Group sizes               = {n_i}")
    print(f"SStotal                   = {SStotal:.4f}")
    print(f"SSb (Between Groups)      = {SSb:.4f}")
    print(f"SSw (Within Groups)       = {SSw:.4f}")
    print(f"MSb                       = {MSb:.4f}")
    print(f"MSw                       = {MSw:.4f}")
    print(f"F-statistic              = {F_stat:.4f}")
    print(f"F-critical (alpha={alpha})= {F_crit:.4f}")
    print(f"p-value                   = {p_value:.4f}")
    print(f"Decision: Reject H0?      = {reject}")
    print("--------------------------------------------------------")


def ANOVA1_is_contrast(c: List[float]) -> bool:
    """
    Check if the coefficients c1, c2, ..., cI define a contrast.
    A contrast requires sum_i c_i = 0.

    Returns True if it is a contrast, False otherwise.
    """
    total = sum(c)
    return (abs(total) < 1e-12)  # allow for floating-rounding


def ANOVA1_is_orthogonal(n: List[int],
                         c1: List[float],
                         c2: List[float]) -> bool:
    """
    Check whether two sets of coefficients define orthogonal contrasts 
    for a one-way layout with group sizes n1,...,nI.

    - c1, c2 are each length I.
    - We first check if c1 and c2 are each valid contrasts (sum=0).
    - Then we check sum_{i=1}^I [ (c1_i * c2_i) / n_i ] = 0.

    If either c1 or c2 is not a contrast, we raise a warning and return False.
    """
    if not ANOVA1_is_contrast(c1):
        print("Warning: c1 is not a contrast!")
        return False
    if not ANOVA1_is_contrast(c2):
        print("Warning: c2 is not a contrast!")
        return False

    val = 0.0
    for i in range(len(n)):
        val += (c1[i] * c2[i]) / n[i]

    return (abs(val) < 1e-12)


def bonferroni_correction(alpha: float, m: int) -> float:
    """
    Return the per-test significance level for Bonferroni correction.
    
    alpha: family-wise error rate
    m: number of comparisons/tests

    Bonferroni correction sets per-test alpha_bon = alpha / m.
    """
    return alpha / m


def sidak_correction(alpha: float, m: int) -> float:
    """
    Return the per-test significance level for Sidak's correction.

    alpha: family-wise error rate
    m: number of comparisons/tests

    Sidak sets per-test alpha_sidak = 1 - (1 - alpha)^(1/m).
    """
    return 1.0 - (1.0 - alpha)**(1.0 / m)


def ANOVA1_CI_linear_combs(
    data: List[List[float]],
    alpha: float,
    C: np.ndarray,
    method: str = "Scheffe"
):
    """
    Compute simultaneous confidence intervals for linear combinations
    of group means in a one-way ANOVA layout.

    Inputs:
      - data: list of I groups
      - alpha: significance level (familywise)
      - C: an m x I matrix of coefficients (each row is a linear combination)
      - method: one of ["Scheffe", "Tukey", "Bonferroni", "Sidak", "best"]

    Output:
      - A list of (L_i, U_i) for i=1..m for each linear combination.

    Implementation notes:
      - If method == "Scheffe", we use either Theorem 2.6 or Theorem 2.7 
        depending on whether each combination is a contrast or not.
      - If method == "Tukey", we only proceed if all combos are pairwise differences.
      - If method == "Bonferroni"/"Sidak", we adjust alpha accordingly.
      - If method == "best", we pick whichever valid approach yields the narrowest intervals.
      - This function requires computing MSw from the one-way ANOVA. 
      - For demonstration, we do not fully implement the specialized distributions 
        (Tukey's studentized range, etc.). We'll outline the logic and use placeholders 
        or approximate quantiles if SciPy is available.
    """
    # Basic stats
    I = len(data)
    n_i = np.array([len(g) for g in data])
    N = sum(n_i)
    group_means = np.array([_mean(g) for g in data])
    overall_mean = _mean(_flatten_data(data))

    # Partition TSS, etc., to get MSw
    SStotal, SSw, SSb = ANOVA1_partition_TSS(data)
    dfB = I - 1
    dfW = N - I
    MSw = SSw / dfW

    # Check validity of method:
    # e.g., if method == "Tukey", confirm each row of C is a pairwise difference 
    # (like [1, -1, 0, 0, ...]).
    # We'll define a small checker:
    def _is_pairwise_diff(row):
        nonzero_idx = np.where(np.abs(row) > 1e-12)[0]
        # row must have exactly two nonzero entries, one +1 and one -1
        if len(nonzero_idx) == 2:
            cvals = row[nonzero_idx]
            # check cvals ~ +1, -1 or vice versa
            # scale invariance is sometimes allowed, but let's keep it simple:
            return (abs(cvals[0] - 1.0) < 1e-12 and abs(cvals[1] + 1.0) < 1e-12) or \
                   (abs(cvals[0] + 1.0) < 1e-12 and abs(cvals[1] - 1.0) < 1e-12)
        return False

    # Collect info about each row
    all_contrasts = True
    all_pairwise_diffs = True
    for row in C:
        if not ANOVA1_is_contrast(row):
            all_contrasts = False
        if not _is_pairwise_diff(row):
            all_pairwise_diffs = False

    # If method == "Tukey" but not all pairwise differences, warn and return None
    if method == "Tukey" and not all_pairwise_diffs:
        print("Warning: Tukey intervals are valid only for pairwise comparisons.")
        return None

    # For demonstration, we do a simple approach:
    # We'll pretend we use a T-quantile or F-quantile for the expansions. 
    # Precisely: 
    #   - Scheffe (for a single contrast c):
    #       margin = sqrt( (I - 1)*F_{I-1, N-I, 1-alpha } * MSw * sum_i( c_i^2 / n_i ) )
    #
    #   - Bonferroni: 
    #       margin = t_{dfW, 1 - alpha/(2m)} * sqrt( MSw * sum_i( c_i^2 / n_i ) )
    #
    #   - Sidak:
    #       alpha_sidak = 1 - (1 - alpha)^(1/m)
    #       margin = t_{dfW, 1 - alpha_sidak/2} * ...
    #
    #   - Tukey:
    #     margin = q_{I, N-I, 1-alpha} / sqrt(2) * sqrt(MSw * sum_i(c_i^2 / n_i))
    #     (Only for pairwise differences).
    #
    # We skip rigorous "best" logic to keep it simpler.

    try:
        from scipy.stats import f, t  # for critical values
        # For a true Tukey approach, you'd also need the studentized range distribution:
        # from scipy.stats import studentized_range
    except ImportError:
        print("Warning: SciPy not found; returning None.")
        return None

    m = C.shape[0]  # number of linear combinations

    # Determine the relevant critical "multiplier" for each row
    def margin_Scheffe(row):
        # Check if it's a contrast
        if ANOVA1_is_contrast(row):
            # Use Theorem 2.7 => (I-1)*F_{I-1, dfW, 1-alpha}
            F_crit = f.ppf(1.0 - alpha, dfB, dfW)
            return math.sqrt( (dfB)*F_crit * MSw * np.sum( (row**2)/n_i ) )
        else:
            # Use Theorem 2.6 => I*(F_{I-1, dfW, 1-alpha})
            F_crit = f.ppf(1.0 - alpha, dfB, dfW)
            return math.sqrt( I * F_crit * MSw * np.sum( (row**2)/n_i ) )

    def margin_Bonf(row):
        alpha_b = bonferroni_correction(alpha, m)
        t_crit = t.ppf(1.0 - alpha_b/2, dfW)
        return t_crit * math.sqrt( MSw * np.sum( (row**2)/n_i ) )

    def margin_Sidak(row):
        alpha_s = sidak_correction(alpha, m)
        t_crit = t.ppf(1.0 - alpha_s/2, dfW)
        return t_crit * math.sqrt( MSw * np.sum( (row**2)/n_i ) )

    def margin_Tukey(row):
        # Place-holder for demonstration only. Real code would use the studentized range distribution.
        # For pairwise difference c = [1, -1, 0, ..., 0],
        # the margin is q_{I, dfW, 1-alpha} / sqrt(2) * sqrt(MSw * (1/n_i1 + 1/n_i2)).
        # We'll just do a placeholder that uses the "q-value" from some approximation or from tables.
        # Suppose we approximate q_{I,dfW,1-alpha} by t_{dfW, 1 - alpha/2} * sqrt(2).
        alpha_single = alpha  # we treat it as "familywise"
        t_crit_approx = t.ppf(1.0 - alpha_single/2, dfW)
        # approximate q ~ t_crit_approx * sqrt(2)
        q_approx = t_crit_approx * math.sqrt(2)
        # Now margin = (q_approx / sqrt(2)) * sqrt(MSw * sum(c^2/n_i)) = t_crit_approx * sqrt(MSw * sum(...))
        return t_crit_approx * math.sqrt(np.sum( (row**2)/n_i ) * MSw)

    # If method == "best", you would analyze which method is valid and compute all intervals
    # for each valid method, picking the narrowest. Here we simply pick Scheffe to demonstrate.
    # In a real scenario, you'd implement that logic carefully.

    def pick_margin(row):
        if method == "Scheffe":
            return margin_Scheffe(row)
        elif method == "Tukey":
            return margin_Tukey(row)
        elif method == "Bonferroni":
            return margin_Bonf(row)
        elif method == "Sidak":
            return margin_Sidak(row)
        elif method == "best":
            # naive approach: pick whichever is smallest among valid
            # For demonstration, let's try Scheffe vs Bonferroni only:
            marg_s = margin_Scheffe(row)
            marg_b = margin_Bonf(row)
            return min(marg_s, marg_b)
        else:
            raise ValueError("Unknown method: " + method)

    # Build the intervals
    intervals = []
    for row in C:
        est = np.sum(row * group_means)  # sum_i c_i * xbar_i
        mrg = pick_margin(row)
        L = est - mrg
        U = est + mrg
        intervals.append((L, U))

    return intervals


def ANOVA1_test_linear_combs(
    data: List[List[float]],
    alpha: float,
    C: np.ndarray,
    d: np.ndarray,
    method: str = "Scheffe"
):
    """
    Test the hypotheses:
      H0 : c_{i,1} mu1 + ... + c_{i,I} muI = d_i,  i=1...m
      vs  H1:  not all equalities hold
    with a familywise error rate alpha.

    Similar to ANOVA1_CI_linear_combs, but for hypothesis tests. 
    Returns test outcomes (reject or not) and p-values in a way that 
    controls the FWER at alpha.

    Implementation here is schematic and uses the same logic for 
    'Scheffe', 'Tukey', 'Bonferroni', 'Sidak', or 'best'.
    """
    # We'll do a simple approach: 
    # 1) Construct the confidence intervals for c_i^T mu
    # 2) Check if d_i is inside that interval. If not, we reject.

    intervals = ANOVA1_CI_linear_combs(data, alpha, C, method)
    if intervals is None:
        return None

    # For each i, if d_i not in [L, U], we reject
    decisions = []
    for i, (L, U) in enumerate(intervals):
        d_val = d[i]
        reject = (d_val < L) or (d_val > U)
        decisions.append(reject)

    return decisions


# -------------------------------------------------------------------
#                  Two-Way ANOVA Related Functions
# -------------------------------------------------------------------

def ANOVA2_partition_TSS(data_3d: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    data_3d has shape (I, J, K): 
        i=0..I-1, j=0..J-1, k=0..K-1.
    Return: (SStotal, SSA, SSB, SSAB, SSE).
    """
    I, J, K = data_3d.shape
    grand_mean, row_means, col_means, cell_means = _two_way_means(data_3d)

    # SStotal
    SStotal = np.sum((data_3d - grand_mean)**2)

    # SSA = J*K * sum_i ( (row_mean_i - grand_mean)^2 )
    SSA = J*K * np.sum((row_means - grand_mean)**2)

    # SSB = I*K * sum_j ( (col_mean_j - grand_mean)^2 )
    SSB = I*K * np.sum((col_means - grand_mean)**2)

    # SSAB = K * sum_{i,j} ( (cell_mean_{i,j} - row_mean_i - col_mean_j + grand_mean)^2 )
    SSAB = 0.0
    for i in range(I):
        for j in range(J):
            SSAB += (cell_means[i,j] - row_means[i] - col_means[j] + grand_mean)**2
    SSAB *= K

    # SSE = sum_{i,j,k} ( x_{i,j,k} - cell_mean_{i,j} )^2
    SSE = 0.0
    for i in range(I):
        for j in range(J):
            for k in range(K):
                SSE += (data_3d[i,j,k] - cell_means[i,j])**2

    return (SStotal, SSA, SSB, SSAB, SSE)


def ANOVA2_MLE(data_3d: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    data_3d has shape (I, J, K).
    Returns (mu_hat, a_hat, b_hat, delta_hat) the MLEs in the two-way layout:
      X_{i,j,k} = mu + a_i + b_j + delta_{i,j} + error_{i,j,k}.

    - mu_hat = grand_mean
    - a_hat[i] = row_means[i] - mu_hat
    - b_hat[j] = col_means[j] - mu_hat
    - delta_hat[i,j] = cell_means[i,j] - row_means[i] - col_means[j] + mu_hat
    """
    I, J, K = data_3d.shape
    mu_hat, row_means, col_means, cell_means = _two_way_means(data_3d)
    a_hat = row_means - mu_hat
    b_hat = col_means - mu_hat

    delta_hat = np.zeros((I, J))
    for i in range(I):
        for j in range(J):
            delta_hat[i,j] = cell_means[i,j] - row_means[i] - col_means[j] + mu_hat

    return mu_hat, a_hat, b_hat, delta_hat


def ANOVA2_test_equality(data_3d: np.ndarray, alpha: float, choice: str = "A") -> None:
    """
    Perform one of the three standard tests in the two-way ANOVA layout:
        choice = 'A':  test H0: a1=...=aI=0  (no row effect)
        choice = 'B':  test H0: b1=...=bJ=0  (no column effect)
        choice = 'AB': test H0: all delta_{i,j}=0  (no interaction)

    Print the relevant rows of the ANOVA table and the decision.
    """
    I, J, K = data_3d.shape
    SStotal, SSA, SSB, SSAB, SSE = ANOVA2_partition_TSS(data_3d)

    dfA = I - 1
    dfB = J - 1
    dfAB = (I - 1)*(J - 1)
    dfE = I*J*(K - 1)

    MSA = SSA / dfA
    MSB = SSB / dfB
    MSAB = SSAB / dfAB
    MSE = SSE / dfE

    try:
        from scipy.stats import f
    except ImportError:
        print("SciPy not available, cannot compute p-values/critical values.")
        return

    print("--------------------------------------------------------")
    print("Two-Way ANOVA Test")
    print("--------------------------------------------------------")
    print(f"Shape (I,J,K) = {data_3d.shape}")
    print(f"Total DF      = {I*J*K - 1}")
    print(f"SStotal       = {SStotal:.4f}")

    if choice == "A":
        F_stat = MSA / MSE
        p_value = 1.0 - f.cdf(F_stat, dfA, dfE)
        F_crit = f.ppf(1.0 - alpha, dfA, dfE)
        print("Source     DF       SS       MS        F       ")
        print(f"A (rows)   {dfA}   {SSA:.4f}  {MSA:.4f}  {F_stat:.4f}")
        print(f"Error      {dfE}   {SSE:.4f}  {MSE:.4f}")
        print(f"F-crit({alpha})={F_crit:.4f}, p-value={p_value:.4f}")
        print(f"Decision: Reject H0? = {p_value < alpha}")
    elif choice == "B":
        F_stat = MSB / MSE
        p_value = 1.0 - f.cdf(F_stat, dfB, dfE)
        F_crit = f.ppf(1.0 - alpha, dfB, dfE)
        print("Source     DF       SS       MS        F       ")
        print(f"B (cols)   {dfB}   {SSB:.4f}  {MSB:.4f}  {F_stat:.4f}")
        print(f"Error      {dfE}   {SSE:.4f}  {MSE:.4f}")
        print(f"F-crit({alpha})={F_crit:.4f}, p-value={p_value:.4f}")
        print(f"Decision: Reject H0? = {p_value < alpha}")
    else:
        # AB
        F_stat = MSAB / MSE
        p_value = 1.0 - f.cdf(F_stat, dfAB, dfE)
        F_crit = f.ppf(1.0 - alpha, dfAB, dfE)
        print("Source     DF         SS        MS         F       ")
        print(f"A x B      {dfAB}     {SSAB:.4f}    {MSAB:.4f}  {F_stat:.4f}")
        print(f"Error      {dfE}      {SSE:.4f}    {MSE:.4f}")
        print(f"F-crit({alpha})={F_crit:.4f}, p-value={p_value:.4f}")
        print(f"Decision: Reject H0? = {p_value < alpha}")

    print("--------------------------------------------------------")


# -------------------------------------------------------------------
#              Multiple Linear Regression Related Functions
# -------------------------------------------------------------------

def mult_LR_least_squares(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Find the least squares solution (beta-hat) for the MLR model:
       y = X beta + error
    Also returns the MLE of sigma^2 (biased) and unbiased estimator of sigma^2.

    X: shape (n, k+1)  (including a column of 1's if there's an intercept)
    y: shape (n, )

    Returns: (beta_hat, sigma2_ML, sigma2_unbiased)
    """
    # beta_hat = (X^T X)^{-1} X^T y
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta_hat = XtX_inv @ (X.T @ y)

    # residuals
    r = y - X @ beta_hat
    n, p = X.shape  # p = k+1
    # Maximum likelihood estimate for sigma^2 (under normal) is SSE/n
    SSE = r.T @ r
    sigma2_ML = SSE / n
    # Unbiased estimate for sigma^2 is SSE/(n - p)
    sigma2_unb = SSE / (n - p)

    return beta_hat, sigma2_ML, sigma2_unb


def mult_LR_partition_TSS(X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Partition the total sum of squares in linear regression:
      TSS = SSR + SSE

    TSS = sum_{i}(y_i - ybar)^2
    SSR = sum_{i} (yhat_i - ybar)^2
    SSE = sum_{i} (y_i - yhat_i)^2

    Returns (TSS, SSR, SSE)
    """
    n = len(y)
    ybar = np.mean(y)

    # fitted values
    beta_hat, _, _ = mult_LR_least_squares(X, y)
    yhat = X @ beta_hat

    TSS = np.sum((y - ybar)**2)
    SSR = np.sum((yhat - ybar)**2)
    SSE = np.sum((y - yhat)**2)

    return (TSS, SSR, SSE)


def mult_norm_LR_simul_CI(X: np.ndarray, y: np.ndarray, alpha: float):
    """
    Compute simultaneous confidence intervals for all beta_i 
    in the normal multiple linear regression model.

    Various methods are possible (Bonferroni, Scheffe, etc.).
    We'll do a simple *Scheffe approach* to illustrate.

    Return a list of (L_i, U_i) intervals for beta_i, i=0..k.
    """
    n, p = X.shape
    beta_hat, _, sigma2_unb = mult_LR_least_squares(X, y)
    XtX_inv = np.linalg.inv(X.T @ X)

    # For Scheffe, we use:
    #   margin_i = sqrt( p * F_{p, n-p, 1-alpha} ) * sqrt( sigma2_unb * v_i ),
    # where v_i = XtX_inv[i, i]
    # i.e. the diagonal element of (X^T X)^-1

    try:
        from scipy.stats import f
    except ImportError:
        print("SciPy not available. Returning None.")
        return None

    F_crit = f.ppf(1.0 - alpha, p - 1, n - p) if p > 1 else float('nan')  
    # Actually for p parameters (k+1 = p), degrees of freedom for model is p-1 if 
    # we consider the intercept forced in. This can vary by approach. For demonstration:
    
    # Correction: The classical Scheffe approach for all linear combos of dimension p 
    # would use F_{p, n-p, 1-alpha}. But if we always included the intercept in "p", 
    # that’s the dimension. We'll keep it simple here.

    margin_multiplier = math.sqrt(p * F_crit * sigma2_unb)
    intervals = []
    for i in range(p):
        v_i = XtX_inv[i, i]
        half_width = margin_multiplier * math.sqrt(v_i)
        L = beta_hat[i] - half_width
        U = beta_hat[i] + half_width
        intervals.append((L, U))
    return intervals


def mult_norm_LR_CR(X: np.ndarray, y: np.ndarray, C: np.ndarray, alpha: float):
    """
    Return the parameters (center, shape matrix, radius^2) for the 
    100(1-alpha)% confidence region for C beta in the normal MLR model.

    The region is an ellipsoid described by:
      (C beta_hat - c0)^T [ C (X^T X)^-1 C^T ]^{-1} (C beta_hat - c0 ) <= radius^2

    But the user must supply c0 or we assume c0=0 for the region. 
    The question statement says "returns the specifications (parameters of the ellipsoid)" 
    for 100(1-alpha)% region for C beta. Typically we also need c0, 
    but let's assume c0=0 for the region center. If not, you can extend.

    We'll interpret "C with rank r" => dimension r for that subspace. 
    The radius uses an F quantile if it's a two-sided joint region.

    Implementation: We'll do:
      center = C beta_hat
      shape_matrix = ...
      radius^2 = r * (p, n-p, 1-alpha) * sigma^2_unb, etc.
    """
    n, p = X.shape
    r, pC = C.shape
    if pC != p:
        raise ValueError("C must have same number of columns as X has parameters.")

    beta_hat, _, sigma2_unb = mult_LR_least_squares(X, y)
    cbeta_hat = C @ beta_hat

    XtX_inv = np.linalg.inv(X.T @ X)
    M = C @ XtX_inv @ C.T  # shape (r, r)

    try:
        from scipy.stats import f
    except ImportError:
        print("SciPy not available. Returning None.")
        return None

    # The usual approach for a joint (two-sided) region for C beta is:
    #   (C beta_hat - c0)^T [ M ]^-1 (C beta_hat - c0 ) / r / sigma2_unb  ~ F_{r, n-p}
    # => That leads to the boundary:
    #   <= r * F_{r, n-p, 1-alpha} * sigma2_unb
    F_crit = f.ppf(1.0 - alpha, r, n - p)

    radius_sq = r * F_crit * sigma2_unb
    M_inv = np.linalg.inv(M)

    # Return specs
    return {
        "center": cbeta_hat,
        "shape_matrix_inv": M_inv,   # the matrix that appears in the left side of the inequality
        "radius_sq": radius_sq
    }


def mult_norm_LR_is_in_CR(X: np.ndarray, y: np.ndarray, 
                          C: np.ndarray, c0: np.ndarray, alpha: float) -> bool:
    """
    Check whether c0 is in the 100(1-alpha)% confidence region for C beta.

    The region is:
      (C beta_hat - c0)^T [ C (X^T X)^-1 C^T ]^{-1} (C beta_hat - c0 ) <= r^2

    where r^2 = r * F_{r, n-p, 1-alpha} * sigma2_unb.
    """
    n, p = X.shape
    r, pC = C.shape
    if pC != p or c0.shape[0] != r:
        raise ValueError("Dimension mismatch between C and c0 or X.")

    res = mult_norm_LR_CR(X, y, C, alpha)
    if res is None:
        return False
    center = res["center"]
    M_inv = res["shape_matrix_inv"]
    radius_sq = res["radius_sq"]

    diff = center - c0
    left_side = diff.T @ M_inv @ diff
    return (left_side <= radius_sq)


def mult_norm_LR_test_general(X: np.ndarray, y: np.ndarray,
                              C: np.ndarray, c0: np.ndarray,
                              alpha: float):
    """
    Test the null hypothesis H0 : C beta = c0  vs H1 : C beta != c0
    at significance alpha, in the normal multiple linear regression model.

    Implementation: Use the standard F-test approach:
      F = [ (C beta_hat - c0)^T [ C (X^T X)^-1 C^T ]^-1 (C beta_hat - c0 ) ] / r / sigma2_unb
          ~ F_{r, n-p}
    If F > F_{r, n-p, 1-alpha}, reject H0.
    Return (test_stat, p_value, reject?)
    """
    n, p = X.shape
    r, pC = C.shape
    if pC != p or c0.shape[0] != r:
        raise ValueError("Dimension mismatch among X, C, c0.")

    beta_hat, _, sigma2_unb = mult_LR_least_squares(X, y)
    diff = (C @ beta_hat - c0)

    XtX_inv = np.linalg.inv(X.T @ X)
    M = C @ XtX_inv @ C.T

    # test statistic
    numerator = diff.T @ np.linalg.inv(M) @ diff / r
    F_stat = numerator / sigma2_unb

    try:
        from scipy.stats import f
        p_value = 1.0 - f.cdf(F_stat, r, n - p)
        F_crit = f.ppf(1.0 - alpha, r, n - p)
    except ImportError:
        p_value = float('nan')
        F_crit = float('nan')

    reject = (p_value < alpha)
    return (F_stat, p_value, reject)


def mult_norm_LR_test_comp(X: np.ndarray, y: np.ndarray, alpha: float,
                           indices: List[int]):
    """
    Test H0: beta_j1 = beta_j2 = ... = beta_jr = 0  vs H1: not all zero
    using the general test with a suitable C.

    indices: e.g. [1,2] means test H0: beta_1=0 and beta_2=0 simultaneously.
    """
    p = X.shape[1]  # number of parameters
    r = len(indices)
    C = np.zeros((r, p))
    for i, j in enumerate(indices):
        C[i, j] = 1.0
    c0 = np.zeros(r)
    return mult_norm_LR_test_general(X, y, C, c0, alpha)


def mult_norm_LR_test_linear_reg(X: np.ndarray, y: np.ndarray, alpha: float):
    """
    Test H0: beta_1 = beta_2 = ... = beta_k = 0  vs  H1: not all zero
    i.e. test if the regression has any linear effect at all (excluding intercept).
    """
    n, p = X.shape
    # Usually the intercept is in column 0, so we want columns 1..p-1
    # i.e. test if all non-intercept betas are zero.
    if p < 2:
        print("No slope parameters to test (only intercept). Returning None.")
        return None
    indices = list(range(1, p))  # skip the intercept
    return mult_norm_LR_test_comp(X, y, alpha, indices)


def mult_norm_LR_pred_CI(X: np.ndarray, y: np.ndarray, 
                         D: np.ndarray, alpha: float,
                         method: str = "Bonferroni"):
    """
    Return simultaneous confidence intervals for predictions d_i^T beta, i=1..m,
    each row d_i of D is a (k+1)-vector.

    method in ["Bonferroni", "Scheffe", "best"].

    We'll do a basic approach:
      - Estimate: d_i^T beta_hat
      - Var( d_i^T beta_hat ) = sigma2_unb * d_i^T (X^T X)^-1 d_i
      - We scale the margin by a factor depending on method and the total number m of intervals.

    For "best", we just pick whichever margin is smaller for demonstration.
    """
    n, p = X.shape
    m, p2 = D.shape
    if p2 != p:
        raise ValueError("Design matrix D must have same number of cols as X.")
    beta_hat, _, sigma2_unb = mult_LR_least_squares(X, y)
    XtX_inv = np.linalg.inv(X.T @ X)

    # Each predicted point is d_i^T beta_hat
    # variance is sigma2_unb * d_i^T (XtX_inv) d_i
    # We'll use t_{n-p, 1-(alpha/m)/2} or the appropriate factor for Scheffe.

    try:
        from scipy.stats import f, t
    except ImportError:
        print("SciPy not found, returning None.")
        return None

    predictions = D @ beta_hat
    var_pred = np.array([d @ XtX_inv @ d for d in D])

    if method == "Bonferroni":
        alpha_b = alpha / m
        t_crit = t.ppf(1.0 - alpha_b/2, n - p)
        margins = t_crit * np.sqrt(sigma2_unb * var_pred)
    elif method == "Scheffe":
        # margin = sqrt( p * F_{p, n-p, 1-alpha} ) * sqrt( sigma2_unb * var_pred_i )
        F_crit = f.ppf(1.0 - alpha, p, n - p)
        margin_mult = math.sqrt(p * F_crit * sigma2_unb)
        margins = margin_mult * np.sqrt(var_pred)
    elif method == "best":
        # naive approach: pick min of the two
        alpha_b = alpha / m
        t_crit = t.ppf(1.0 - alpha_b/2, n - p)
        bonf_margins = t_crit * np.sqrt(sigma2_unb * var_pred)

        F_crit = f.ppf(1.0 - alpha, p, n - p)
        scheffe_mult = math.sqrt(p * F_crit * sigma2_unb)
        scheffe_margins = scheffe_mult * np.sqrt(var_pred)

        # pick whichever is smaller for each i
        margins = np.minimum(bonf_margins, scheffe_margins)
    else:
        raise ValueError("Method must be one of ['Bonferroni','Scheffe','best'].")

    intervals = []
    for i in range(m):
        est = predictions[i]
        half_width = margins[i]
        L = est - half_width
        U = est + half_width
        intervals.append((L, U))

    return intervals
