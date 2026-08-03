# Experiment B -- Per-user Delta EER (Transformer - LST-Former)

Positive Delta EER => LST-Former better for that user.
Restricted to the **n=81** users present in every model x N cell.

## N = 25

- Mean Delta EER = **0.0426** (median 0.0113)
- LST-Former better for **61.7%** of users (50/81)
- Transformer better for **37.0%** (30/81)
- Top 10% of winners account for **32.5%** of total positive Delta EER mass; top 20% -> **58.2%**
- One-sample t-test (mean Delta > 0): p=0.003282 (sig)
- Wilcoxon (Delta > 0): p=0.00164 (sig)
- Mean AUC Delta (LST-F - Trans) = 0.0477
- Mean recon-gap Delta (LST-F - Trans) = -7.4999

**Reading:** Advantage looks **concentrated** in a subset of users (top quintile of winners carries >55% of positive mass).

## N = 50

- Mean Delta EER = **0.0342** (median 0.0102)
- LST-Former better for **65.4%** of users (53/81)
- Transformer better for **32.1%** (26/81)
- Top 10% of winners account for **39.9%** of total positive Delta EER mass; top 20% -> **64.5%**
- One-sample t-test (mean Delta > 0): p=0.01404 (sig)
- Wilcoxon (Delta > 0): p=0.0009412 (sig)
- Mean AUC Delta (LST-F - Trans) = 0.0373
- Mean recon-gap Delta (LST-F - Trans) = 3.8140

**Reading:** Advantage looks **concentrated** in a subset of users (top quintile of winners carries >55% of positive mass).

## N = 100

- Mean Delta EER = **0.0134** (median 0.0055)
- LST-Former better for **56.8%** of users (46/81)
- Transformer better for **42.0%** (34/81)
- Top 10% of winners account for **35.6%** of total positive Delta EER mass; top 20% -> **65.2%**
- One-sample t-test (mean Delta > 0): p=0.3023 (n.s.)
- Wilcoxon (Delta > 0): p=0.1089 (n.s.)
- Mean AUC Delta (LST-F - Trans) = 0.0116
- Mean recon-gap Delta (LST-F - Trans) = -6.2190

**Reading:** Advantage looks **concentrated** in a subset of users (top quintile of winners carries >55% of positive mass).

## Bottom line for the thesis

If % of users helped is high *and* top-quintile share of positive mass
is moderate, the N=25/50 significance is a **broad** effect -- safer to
claim as architectural. If most users are near zero and a few large
winners drive the mean, frame the result as **heterogeneous / user-
dependent**, not a uniform superiority claim.
