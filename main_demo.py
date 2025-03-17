"""
main_demo.py

This script demonstrates usage of the functions in anova_linreg_toolbox.py
with some simple example data. 
"""

import numpy as np
from anova_linreg_toolbox import (
    ANOVA1_partition_TSS,
    ANOVA1_test_equality,
    ANOVA1_is_contrast,
    ANOVA1_is_orthogonal,
    bonferroni_correction,
    sidak_correction,
    ANOVA1_CI_linear_combs,
    ANOVA1_test_linear_combs,
    ANOVA2_partition_TSS,
    ANOVA2_MLE,
    ANOVA2_test_equality,
    mult_LR_least_squares,
    mult_LR_partition_TSS,
    mult_norm_LR_simul_CI,
    mult_norm_LR_CR,
    mult_norm_LR_is_in_CR,
    mult_norm_LR_test_general,
    mult_norm_LR_test_comp,
    mult_norm_LR_test_linear_reg,
    mult_norm_LR_pred_CI
)

def main():
    # ----------------------------------------------------------------
    # 1. Demonstrate One-Way ANOVA
    # ----------------------------------------------------------------
    print("=== One-Way ANOVA Demo ===")
    data_1way = [
        [4.2, 5.1, 4.8, 5.0],
        [5.5, 5.0, 5.2, 5.1, 5.3],
        [6.0, 5.9, 6.1, 5.8]
    ]
    # 1a) Partition TSS
    SStotal, SSw, SSb = ANOVA1_partition_TSS(data_1way)
    print(f"SStotal={SStotal:.4f}, SSw={SSw:.4f}, SSb={SSb:.4f}")

    # 1b) ANOVA test
    ANOVA1_test_equality(data_1way, alpha=0.05)

    # 1c) Contrasts
    c = [1, -1, 0]   # Example contrast?
    print(f"Is {c} a contrast? {ANOVA1_is_contrast(c)}")

    # 1d) Orthogonal
    n = [len(g) for g in data_1way]
    c1 = [1, -1, 0]
    c2 = [1, 0, -1]
    ortho = ANOVA1_is_orthogonal(n, c1, c2)
    print(f"c1 and c2 orthogonal? {ortho}")

    # 1e) Corrections
    alpha = 0.05
    m = 3
    alpha_b = bonferroni_correction(alpha, m)
    alpha_s = sidak_correction(alpha, m)
    print(f"Bonferroni alpha={alpha_b:.4f}, Sidak alpha={alpha_s:.4f}")

    # 1f) Confidence intervals for linear combos
    import numpy as np
    C = np.array([
        [1, -1, 0],  # compare group1 and group2
        [0, 1, -1]   # compare group2 and group3
    ])
    intervals = ANOVA1_CI_linear_combs(data_1way, 0.05, C, method="Bonferroni")
    print("Simultaneous CIs (Bonferroni) for c^T mu:")
    for i, (L, U) in enumerate(intervals):
        print(f"Combination {i}, L={L:.4f}, U={U:.4f}")

    # 1g) Testing linear combos
    d = np.array([0.0, 0.0])  # hypothesize difference = 0
    decisions = ANOVA1_test_linear_combs(data_1way, 0.05, C, d, method="Bonferroni")
    print(f"Reject each linear combo = {decisions}")


    # ----------------------------------------------------------------
    # 2. Demonstrate Two-Way ANOVA
    # ----------------------------------------------------------------
    print("\n=== Two-Way ANOVA Demo ===")
    # Suppose we have I=2, J=3, K=4 data
    data_2way = np.array([
        [
            [4.1, 4.3, 4.0, 4.2],
            [5.1, 5.0, 4.9, 5.2],
            [5.9, 5.8, 6.0, 6.1]
        ],
        [
            [3.9, 4.0, 3.8, 4.1],
            [5.2, 5.1, 5.3, 5.0],
            [6.2, 6.3, 6.1, 6.0]
        ],
    ])
    # 2a) Partition TSS
    SStotal, SSA, SSB, SSAB, SSE = ANOVA2_partition_TSS(data_2way)
    print(f"SStotal={SStotal:.4f}, SSA={SSA:.4f}, SSB={SSB:.4f}, SSAB={SSAB:.4f}, SSE={SSE:.4f}")

    # 2b) MLE
    mu_hat, a_hat, b_hat, delta_hat = ANOVA2_MLE(data_2way)
    print(f"mu_hat={mu_hat:.4f}")
    print(f"a_hat={a_hat}")
    print(f"b_hat={b_hat}")
    print(f"delta_hat=\n{delta_hat}")

    # 2c) Test equality (e.g. row effect)
    ANOVA2_test_equality(data_2way, 0.05, choice="A")
    # Could also do "B" or "AB"


    # ----------------------------------------------------------------
    # 3. Demonstrate Multiple Linear Regression
    # ----------------------------------------------------------------
    print("\n=== Multiple Linear Regression Demo ===")
    # Generate a small design matrix X and response y
    # Suppose we have 8 observations, 1 intercept + 2 predictors => p=3
    X = np.array([
        [1, 0.0, 2.1],
        [1, 1.0, 2.3],
        [1, 2.0, 2.7],
        [1, 3.0, 3.1],
        [1, 4.0, 3.8],
        [1, 5.0, 4.0],
        [1, 6.0, 4.2],
        [1, 7.0, 4.4],
    ])
    y = np.array([2.0, 2.3, 2.4, 2.6, 3.0, 3.4, 3.7, 4.0])

    # 3a) Least squares
    beta_hat, sigma2_ML, sigma2_unb = mult_LR_least_squares(X, y)
    print(f"beta_hat={beta_hat}, sigma2_ML={sigma2_ML:.4f}, sigma2_unb={sigma2_unb:.4f}")

    # 3b) Partition TSS
    TSS, SSR, SSE = mult_LR_partition_TSS(X, y)
    print(f"TSS={TSS:.4f}, SSR={SSR:.4f}, SSE={SSE:.4f}")

    # 3c) Simultaneous CI for betas
    intervals = mult_norm_LR_simul_CI(X, y, alpha=0.05)
    print("Scheffe CIs for betas:")
    if intervals is not None:
        for i, (L, U) in enumerate(intervals):
            print(f"beta_{i} in [{L:.4f}, {U:.4f}]")

    # 3d) Confidence region for some linear combo C beta
    # e.g. we want to see region for [0, 1, -1]*beta
    C = np.array([[0, 1, -1]])
    region_specs = mult_norm_LR_CR(X, y, C, alpha=0.05)
    print("Confidence region specs for C beta:")
    if region_specs is not None:
        print(region_specs)

    # 3e) Check if a particular c0 is in the region
    c0 = np.array([0.0])  # test if the difference of those 2 coefficients is 0
    in_region = mult_norm_LR_is_in_CR(X, y, C, c0, alpha=0.05)
    print(f"Is c0={c0} in CR? {in_region}")

    # 3f) Test general linear hypothesis
    F_stat, p_value, reject = mult_norm_LR_test_general(X, y, C, c0, alpha=0.05)
    print(f"Test H0: (beta2 - beta1)=0 => F={F_stat:.4f}, p={p_value:.4f}, reject?={reject}")

    # 3g) Test if certain betas are zero
    # e.g. test H0: beta_1=0 (the second param, in the indexing scheme)
    test_result = mult_norm_LR_test_comp(X, y, 0.05, [1])
    print(f"Test H0: beta_1=0 => F={test_result[0]:.4f}, p={test_result[1]:.4f}, reject?={test_result[2]}")

    # 3h) Test H0: no linear regression at all
    # i.e. test all slopes except intercept
    test_linreg_result = mult_norm_LR_test_linear_reg(X, y, 0.05)
    if test_linreg_result is not None:
        print("Test H0: no linear regression =>", test_linreg_result)

    # 3i) Prediction intervals
    D = np.array([
        [1, 2.5, 3.0],
        [1, 4.5, 3.5]
    ])
    pred_intervals = mult_norm_LR_pred_CI(X, y, D, 0.05, method="best")
    print("Prediction intervals for new points:")
    if pred_intervals is not None:
        for i, (L, U) in enumerate(pred_intervals):
            print(f"Pred {i}: [{L:.4f}, {U:.4f}]")

if __name__ == "__main__":
    main()
