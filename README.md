===========================================================
ANOVA & Linear Regression Toolbox (Python)
===========================================================

Overview
--------
This toolbox provides Python functions for common one-way 
and two-way ANOVA analyses, as well as multiple linear 
regression procedures with normal errors. 

Contents
--------
1) anova_linreg_toolbox.py
   - A collection of functions for:
       - One-Way ANOVA 
         (Partition sums of squares, F-test, contrasts, 
          orthogonality, multiple comparison adjustments, 
          confidence intervals and tests for linear combinations)
       - Two-Way ANOVA
         (Partition sums of squares, MLE of parameters, 
          test row effect, column effect, interaction)
       - Multiple Linear Regression
         (Least-squares estimates, partition TSS/SSR/SSE, 
          simultaneous confidence intervals, 
          confidence regions for linear combos, 
          general linear hypothesis testing, 
          prediction intervals)

2) main_demo.py
   - Demonstrates each function in action using small 
     example datasets. You can run it directly:
       $ python main_demo.py
   - The script prints all results to screen.

3) README.txt
   - This file.

Requirements
------------
- Python 3.x
- NumPy (for arrays and linear algebra)
- Optional: SciPy for F-distribution, t-distribution,
  etc. If SciPy is missing, you will see warnings and 
  certain functions won't be able to compute p-values 
  or critical values.

How to Use
----------
1. Place anova_linreg_toolbox.py in your working directory,
   along with main_demo.py and README.txt.
2. In your own scripts, you can import from 
   anova_linreg_toolbox, for example:
   
       from anova_linreg_toolbox import ANOVA1_test_equality

3. See main_demo.py for usage examples.

Notes
-----
- Some methods (like Tukey) require specialized distributions 
  (studentized range). If SciPy is unavailable, or if the 
  distribution is not directly included, the code outlines 
  placeholders or approximations.
- The code is written for educational/demo purposes, not 
  for high-efficiency large-scale computing.
- Pay attention to the shape and structure of your data. 
  One-way ANOVA expects a list-of-lists, two-way ANOVA 
  expects a 3D NumPy array, and regression expects 
  (X,y) as NumPy arrays.
