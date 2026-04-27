# Background Math: MinHash LSH Accuracy

This document walks through how MinHash LSH makes deduplication decisions,
where errors can occur, and how to read the confusion matrix. 
As a reminder, this library/repository does not modify existing MinHash LSH algorithm at all.
This doc is just for reference.

Reference: Leskovec, Rajaraman & Ullman, *Mining of Massive Datasets*, Chapter 3.

---

## Terminology
```
                        Predicted
                    Dup          Not Dup
              ┌────────────┬────────────┐
Actual  Dup   │  TP        │  FN        │  
              ├────────────┼────────────┤   
       Not    │  FP        │   TN       │  
              └────────────┴────────────┘   

- Catch rate (=recall): of real duplicates, how many did we catch
    - ex) 97 / 100  = 97%
- Accuracy rate (=precision): of pairs we flagged, how many are real duplicates
    - ex) 97 / 109  = 89%

```

## Overview: Two Independent Filters

The pipeline has two sequential stages, each with its own source of error:


Document Pair
-> Stage 1: LSH Candidates generate (S-curve probability)
   - what it does: Do their band hashes collide? Purely probabilistic, no threshold input
              True duplicates silently missed
   - related confusion matrix: false negatives
     
-> candidates pair generated

-> Stage 2: Signature Verify (estimateSimilarity)
   
   - what it does: Compare 64/128 MinHash values. User threshold applied here (e.g., 0.9)
              ±StandardError around true similarity
   - related confusion matrix impact:
      - FP is affected : true Jaccard 0.87, noise pushes estimate to 0.91
           -> passes threshold -> false positive
      - FN is also affected : true Jaccard 0.91, noise pushes estimate to 0.87
           -> fails threshold -> false negative
      - TP is also affected because a pair with true 0.92 that estimates at 0.95 still passes, but the estimated value is wrong. It doesn't change the outcome, but the reported similarity is noisy.
      - TN is unaffected. pairs at 0.5 estimated at 0.53 still fail. Noise at low similarity is too small to reach 0.9.
-> pass threshold,  Flagged as Duplicate


These two stages are **independent**. The S-curve controls which pairs are even
considered. The signature comparison controls which candidates are kept.

---

## 1.Generate LSH Candidates details
The S-Curve: LSH Candidacy Probability

The probability that two documents become candidates (their band hashes collide
in at least one band) is:

```
P(candidate) = 1 - (1 - s^r)^b
```

where:
- `s` = true Jaccard similarity
- `r` = rows per band (hashes per band)
- `b` = number of bands
- `r × b` = total number of hashes
    - ex) 64

### How to read this formula

**Inner term: `s^r`**
The probability that all `r` hashes in a single band match.
Each hash matches independently with probability `s`,
so all `r` matching = `s^r`.

**Middle term: `1 - s^r`**
The probability that a single band does NOT match.

**Outer term: `(1 - s^r)^b`**
The probability that NONE of the `b` bands match. This means we miss generating candidate pair when they are truely near duplicates.

**Final: `1 - (1 - s^r)^b`**
The probability that AT LEAST ONE band matches -> candidate pair.

### Example with r=8, b=8 (64 hashes)

Here this table is independent of user defined threshold of similarity.
| True Jaccard | s^r (one band matches) | (1-s^r)^8 (no band matches) | P(candidate) |
|-------------|----------------------|---------------------------|-------------|
| 0.5         | 0.5^8 = 0.0039       | 0.9961^8 = 0.9690         | 0.04%       |
| 0.7         | 0.7^8 = 0.0576       | 0.9424^8 = 0.6234         | 5.6%        |
| 0.8         | 0.8^8 = 0.1678       | 0.8322^8 = 0.2252         | 33%         |
| 0.9         | 0.9^8 = 0.4305       | 0.5695^8 = 0.0168         | 83%         |
| 0.95        | 0.95^8 = 0.6634      | 0.3366^8 = 0.0002         | 98%         |
| 1.0         | 1.0^8 = 1.0          | 0.0^8 = 0.0               | 100%        |

This S-curve means the pipeline catches most true duplicates (high Catch rate above 0.9)
while filtering out the vast majority of non-duplicate comparisons.


### How bands and rows affect the S-curve

For a fixed number of hashes (`k = r × b`), you choose how to allocate them:

```
More rows per band (higher r):
  -> stricter per-band matching (s^r shrinks faster)
  -> S-curve shifts RIGHT (fewer candidates, fewer false positives)
  -> but also fewer true positives at moderate similarity

More bands (higher b):
  -> more chances to collide ((1-s^r)^b shrinks faster)
  -> S-curve shifts LEFT (more candidates, higher Catch rate)
  -> but also more false candidates to verify
```

Example with 64 total hashes allocated differently:

| Config     | P(candidate) at s=0.9 |
|------------|----------------------|
| r=4, b=16  | 99.8%                |
| r=8, b=8   | 83%                  |
| r=16, b=4  | 33%                  |

- **r=4, b=16**: aggressive Catch rate, it reduces missing true near duplicates but increases false positive.
    - standard error is fixed by the number of MinHash samples. catching more docs as candidates with fixed standard error increases false positive
- **r=8, b=8**: balanced tradeoff (common default)
- **r=16, b=4**: very strict, misses many true duplicates

The "ideal" S-curve is a vertical step function at your target threshold. In practice, increasing total hashes (k) makes the curve steeper but cannot make it vertical.

### What about the 17% missed at P(candidate) at true similarity=0.9?

With r=8, b=8: a pair with true Jaccard = 0.9 has a 17% chance of never
becoming a candidate. These pairs are **gone forever** — they never enter
Step 4 and are never compared.

This is the fundamental tradeoff of LSH: we accept some Catch rate loss
to avoid O(n²) comparisons. At 2.5B documents, 83% Catch rate on near-duplicates
is far better than 100% Catch rate on nothing (because the computation is infeasible).

---

## Stage 2: Signature Verify - Standard Error of MinHash

MinHash estimates Jaccard similarity by comparing k hash values:

```
estimated_jaccard = count(sig1[i] == sig2[i]) / len(sig1)
```

The standard error of this estimate is:

```
SE = √(s(1-s) / k)
```

where `s` = true Jaccard similarity, `k` = num_hashes.

| num_hashes | SE at s=0.9 | 95% confidence interval |
|------------|-------------|------------------------|
| 64         | 0.0375      | 0.9 ± 0.075 -> [0.825, 0.975] |
| 128        | 0.0265      | 0.9 ± 0.053 -> [0.847, 0.953] |

More hashes = tighter estimate. This matters at the threshold boundary —
a pair with true Jaccard 0.91 might estimate at 0.87 with 64 hashes
but is less likely to with 128 hashes.

**Standard error only affects Stage 2** (signature comparison).
It has no effect on Stage 1 (LSH candidacy).

---

## Confusion Matrix Example

Using dummy numbers, we walk through the confusion matrics.
 Note that this is just to illustrates how generate LSH Candidates stage and standard error affects the confusion matrix.

### Setup

To understand end-to-end accuracy, we trace 1,000 pairs through both stages.
The pairs are evenly distributed: 100 pairs per similarity bucket.

We define "true duplicate" as true Jaccard ≥ 0.9 (matching the user threshold).

Config: 64 hashes, r=8, b=8, threshold=0.9.

### Step-by-step trace

```
Similarity  Pairs  Stage 1: LSH     Stage 2: Verify   Predicted
Bucket      In     Candidates       (estimate ≥ 0.9)   as Dup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
0.00-0.10   100        0                 0                0
0.10-0.20   100        0                 0                0
0.20-0.30   100        0                 0                0
0.30-0.40   100        0                 0                0
0.40-0.50   100        0                 0                0
0.50-0.60   100        7                 0                0
0.60-0.70   100       23                 0                0
0.70-0.80   100       57                 0                0
0.80-0.90   100       92                12               12 <- FP
0.90-1.00   100      ~97                97               97 <- TP
```

**Reading each column:**

- **Pairs In**: 100 pairs at each similarity level
- **Stage 1 (LSH)**: how many become candidates (S-curve)
- **Stage 2 (Verify)**: of those candidates, how many estimate ≥ 0.9
- **Predicted as Dup**: final output

### Where errors happen

**False Positives (12 pairs from the 0.80-0.90 bucket):**
These pairs have true Jaccard ~0.85. The S-curve let ~92 of them through
as candidates (high probability at 0.85). Then estimation noise pushed
~12 of those estimates above the 0.9 threshold.

Example: true Jaccard = 0.87, but MinHash estimates 0.91 -> passes threshold.

**False Negatives (3 pairs from the 0.90-1.00 bucket):**
These come from both stages:
- ~2 pairs: S-curve missed them (band hashes never collided, gone forever)
- ~1 pair: became a candidate but estimated 0.88 due to noise -> failed threshold

Example: true Jaccard = 0.91, but no band matched -> never compared (S-curve loss)
Example: true Jaccard = 0.90, estimated at 0.87 due to standard error

### Confusion matrix

```
                        Predicted
                    Dup          Not Dup
              ┌────────────┬────────────┐
Actual  Dup   │  97  (TP)  │   3  (FN)  │
              ├────────────┼────────────┤
       Not    │  12  (FP)  │ 888  (TN)  │
              └────────────┴────────────┘

precision (Accuracy rate):  97 / 109  = 89%
recall (Catch rate):     97 / 100  = 97%

F1 score = 2 × (precision × recall) / (precision + recall)
   = 2 × (0.89 × 0.97) / (0.89 + 0.97)
   = 93%
```

### Why Accuracy rate is better in practice

The 12 false positives all come from the 0.80-0.90 bucket — pairs with
true Jaccard ~0.85. Although this walkthrough is imaginary numbers, through benchmark, we had a similar case.
In manual validation of our pipeline, these pairs
are consistently template-heavy pages (e-commerce sites, blog archives,
404 pages) with identical site chrome and minimal unique content.

For LLM training, deduplicating these pages is arguably **correct** —
they add almost no unique information to the training corpus.

If we redefine "true duplicate" as "should be removed for LLM training,"
Accuracy rate approaches 100%.

---

## Summary

| Source of error     | Stage     | Effect           | Mitigation                        |
|---------------------|-----------|-----------------|-----------------------------------|
| S-curve catch miss  | Stage 1   | False negative   | Increase bands (b) for more Catch rate |
| Estimation noise    | Stage 2   | False positive   | Increase total hashes (k)          |
| Estimation noise    | Stage 2   | False negative   | Increase total hashes (k)          |
| Threshold boundary  | Stage 2   | Both FP and FN   | Set threshold away from dense region |

The system is designed to **favor Accuracy rate over Catch rate**. For LLM training
data curation, falsely merging unique content is worse than keeping a few
duplicates. The pipeline's Accuracy rate is validated at 100% across manual
inspection of samples at 1 WET and 1,000 WET file scales.

---

## References

- Broder, A. (1997). *On the Resemblance and Containment of Documents.*
- Leskovec, J., Rajaraman, A., & Ullman, J. *Mining of Massive Datasets*, Chapter 3.
  Freely available at http://www.mmds.org