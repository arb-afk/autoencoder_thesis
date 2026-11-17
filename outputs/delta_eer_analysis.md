# Experiment B -- Per-user Delta EER (Transformer - LST-Former)

Positive Delta EER => LST-Former better for that user.
Restricted to the **n=81** users present in every model x N cell.

## N = 25

- Mean Delta EER = **0.0179** (median 0.0079)
- LST-Former better for **58.0%** of users (47/81)
- Transformer better for **40.7%** (33/81)
- Top 10% of winners account for **40.1%** of total positive Delta EER mass; top 20% -> **61.2%**
- One-sample t-test (mean Delta > 0): p=0.07462 (n.s.)
- Wilcoxon (Delta > 0): p=0.02518 (sig)
- Mean AUC Delta (LST-F - Trans) = 0.0300
- Mean recon-gap Delta (LST-F - Trans) = 5751.4064

**Reading:** Advantage looks **concentrated** in a subset of users (top quintile of winners carries >55% of positive mass).

## N = 50

- Mean Delta EER = **0.0182** (median 0.0081)
- LST-Former better for **63.0%** of users (51/81)
- Transformer better for **35.8%** (29/81)
- Top 10% of winners account for **43.5%** of total positive Delta EER mass; top 20% -> **68.8%**
- One-sample t-test (mean Delta > 0): p=0.1155 (n.s.)
- Wilcoxon (Delta > 0): p=0.02518 (sig)
- Mean AUC Delta (LST-F - Trans) = 0.0343
- Mean recon-gap Delta (LST-F - Trans) = 4794.0862

**Reading:** Advantage looks **concentrated** in a subset of users (top quintile of winners carries >55% of positive mass).

## N = 100

- Mean Delta EER = **0.0117** (median 0.0029)
- LST-Former better for **53.1%** of users (43/81)
- Transformer better for **46.9%** (38/81)
- Top 10% of winners account for **43.8%** of total positive Delta EER mass; top 20% -> **62.2%**
- One-sample t-test (mean Delta > 0): p=0.2471 (n.s.)
- Wilcoxon (Delta > 0): p=0.09892 (n.s.)
- Mean AUC Delta (LST-F - Trans) = 0.0131
- Mean recon-gap Delta (LST-F - Trans) = -3091.0150

**Reading:** Advantage looks **concentrated** in a subset of users (top quintile of winners carries >55% of positive mass).

## Bottom line for the thesis

If % of users helped is high *and* top-quintile share of positive mass
is moderate, the N=25/50 significance is a **broad** effect -- safer to
claim as architectural. If most users are near zero and a few large
winners drive the mean, frame the result as **heterogeneous / user-
dependent**, not a uniform superiority claim.
