# Competition Rules and Terms

**Last Updated**: 2026-01-03

---

## Table of Contents

1. [Overview](#overview)
2. [Prize Information](#prize-information)
3. [Eligibility](#eligibility)
4. [Evaluation Criteria](#evaluation-criteria)
5. [Submission Requirements](#submission-requirements)
6. [Constraints and Rules](#constraints-and-rules)
7. [Timeline](#timeline)
8. [Evaluation Process](#evaluation-process)
9. [Winner Selection](#winner-selection)
10. [Terms and Conditions](#terms-and-conditions)

---

## Overview

The Macro Placement Challenge is a competition to develop better algorithms for VLSI macro placement. Participants compete to beat state-of-the-art methods from Andrew Kahng's influential paper ["Assessment of Reinforcement Learning for Macro Placement"](https://arxiv.org/abs/2302.11014).

**Goal**: Develop a macro placement algorithm that beats the **Circuit Training** baseline on aggregate across all benchmarks.

---

## Prize Information

### Prize Amount

**$20,000 USD** (Twenty Thousand US Dollars)

### Critical Prize Eligibility Requirement

⚠️ **THE PRIZE IS ONLY AWARDED IF YOU BEAT THE CIRCUIT TRAINING BASELINE**

Specifically:
- Your submission must achieve an **aggregate score > 0**
- The aggregate score measures improvement over Circuit Training (Google's RL-based method)
- If no submission beats the baseline, **NO PRIZE WILL BE AWARDED**

### What is the Circuit Training Baseline?

Circuit Training is the best-performing method from the Kahng paper (ISPD 2023). It uses:
- Graph Neural Networks (GNNs) for feature learning
- Reinforcement Learning (RL) for placement decisions
- Google's extensive computational resources

To win the prize, your submission must demonstrate **measurable improvement** over this method on our benchmark suite.

### Aggregate Score Calculation

```python
# Per-benchmark improvement
improvement_i = (CT_baseline_cost_i - your_cost_i) / CT_baseline_cost_i * 100

# Aggregate score = geometric mean across all valid benchmarks
aggregate_score = (∏ improvement_i) ^ (1/N)

# Prize eligibility
if aggregate_score > 0:
    eligible = True  # You beat Circuit Training on average
else:
    eligible = False  # Prize NOT awarded
```

**Example**:
- Benchmark A: +5% improvement (you beat CT by 5%)
- Benchmark B: +3% improvement
- Benchmark C: -2% improvement (CT is better on this one)
- Aggregate: (1.05 × 1.03 × 0.98)^(1/3) ≈ 1.019 = +1.9% → **Prize Eligible** ✓

---

## Eligibility

### Who Can Participate?

The competition is open to:
- ✅ Individuals
- ✅ Teams (prize will be split equally among team members)
- ✅ Academic researchers
- ✅ Industry professionals
- ✅ Students

**Restrictions**:
- ❌ Employees of Partcl or affiliated organizations involved in organizing the competition
- ❌ Individuals in countries subject to US trade restrictions

### Team Participation

- Teams must designate a **team leader** for communication
- Maximum team size: **5 members**
- Team members can be added/removed before the submission deadline
- Prize will be distributed equally among team members (unless otherwise specified)

---

## Evaluation Criteria

### Metrics

Submissions are evaluated using a **proxy cost** that combines three metrics:

```
proxy_cost = 0.5 × wirelength_cost + 0.25 × density_cost + 0.25 × congestion_cost
```

#### 1. Wirelength Cost (50% weight)

Measures total wire length using **Half-Perimeter Wirelength (HPWL)**:

```python
# For each net, compute bounding box half-perimeter
HPWL_net = (max_x - min_x) + (max_y - min_y)

# Weighted sum across all nets
wirelength = Σ (weight_i × HPWL_i)

# Normalize by canvas perimeter
wirelength_cost = wirelength / (2 × (canvas_width + canvas_height))
```

#### 2. Density Cost (25% weight)

Measures how evenly macros are distributed:

```python
# Divide canvas into 32×32 grid
# Compute macro area in each cell
# Penalize high-density regions

density_cost = peak_density / target_density
```

#### 3. Congestion Cost (25% weight)

Estimates routing congestion using simplified routing model:

```python
# Route each net using bounding box approximation
# Count routing demand in each grid cell
# Penalize congested regions

congestion_cost = Σ max(0, demand_i - capacity_i) / total_demand
```

### Improvement Calculation

For each benchmark:

```python
improvement = (baseline_cost - your_cost) / baseline_cost × 100%
```

- **Positive improvement** = your placer is better
- **Negative improvement** = baseline is better
- **0% improvement** = tie

---

## Submission Requirements

### Required Files

1. **placer.py** (required)
   - Must contain a class with `.place(circuit_data)` method
   - Method signature:
     ```python
     def place(self, circuit_data: CircuitTensorData) -> torch.Tensor:
         """
         Returns: [num_macros, 2] tensor of (x, y) coordinates
         """
     ```

2. **requirements.txt** (required)
   - List all Python dependencies
   - Must be installable via `pip install -r requirements.txt`
   - Allowed: PyTorch, NumPy, SciPy, PyTorch Geometric, scikit-learn, etc.
   - Not allowed: External APIs, proprietary software

3. **README.md** (required)
   - Brief description of your approach (1-2 paragraphs)
   - Any special instructions for running your code
   - List team members and affiliations

### Optional Files

- Pre-trained model weights (if using ML)
- Supporting code modules
- Documentation

### File Size Limits

- Maximum submission size: **1 GB**
- If using pre-trained models, weights must be included in submission

---

## Constraints and Rules

### Placement Constraints

All placements must satisfy:

#### 1. No Overlaps

Macros cannot overlap each other. For two macros i and j:

```python
# Bounding boxes must not overlap
not (right_i <= left_j or right_j <= left_i or
     top_i <= bottom_j or top_j <= bottom_i)
```

#### 2. Boundary Constraints

All macros must be fully within the canvas:

```python
# For macro with center (x, y) and size (w, h)
x - w/2 >= 0                  # Left edge
x + w/2 <= canvas_width       # Right edge
y - h/2 >= 0                  # Bottom edge
y + h/2 <= canvas_height      # Top edge
```

#### 3. Respect Blockages

Macros cannot be placed in blockage regions (if specified).

### Runtime Constraints

- **Timeout**: 1 hour (3600 seconds) per benchmark
- Submissions that exceed timeout will be marked as failed for that benchmark
- Failed benchmarks are excluded from aggregate scoring

### Code Constraints

#### Allowed

✅ Any PyTorch-based algorithm (GNNs, RL, optimization, etc.)
✅ Standard Python libraries (NumPy, SciPy, etc.)
✅ Pre-trained models (must be included in submission)
✅ Random seeds for reproducibility
✅ Multi-threading/GPU usage

#### Not Allowed

❌ Modifying evaluation functions or metrics
❌ Using external APIs or services
❌ Hardcoding placements for specific benchmarks
❌ Accessing hidden test benchmarks before evaluation
❌ Exploiting bugs in the evaluation harness (report them instead!)

### Fair Play

- Submissions must be **your own work**
- You may use publicly available libraries and papers
- You must **cite** any external code or ideas used
- **Plagiarism or cheating will result in disqualification**

---

## Timeline

### Competition Phases

#### Phase 1: Beta Launch (Current)

- Public benchmarks available
- Evaluation harness ready
- Participants can submit and test
- Leaderboard is public

#### Phase 2: Competition Period (TBD)

- Competition officially opens
- Submissions accepted
- Leaderboard updates in real-time
- Hidden benchmarks prepared (not revealed)

#### Phase 3: Final Evaluation (TBD)

- Submission deadline
- Final evaluation on public + hidden benchmarks
- Winner verification
- Prize awarded (if eligible)

**Note**: Specific dates will be announced when the competition officially launches.

---

## Evaluation Process

### Public Benchmarks

Available now for development and testing:

- **ariane133**: 133 macros, 757 nets
- **nvdla_asap7**: 128 macros, 2308 nets
- **blackparrot_asap7**: 220 macros, 1892 nets
- **mempool_asap7**: 324 macros, 422 nets
- Plus 3 toy benchmarks for quick testing

You can evaluate your submission anytime:

```bash
python scripts/evaluate_submission.py \
    --submission path/to/your/placer.py \
    --split public
```

### Hidden Test Cases

For final evaluation, we will use 3-5 **hidden benchmarks** that are:
- Not available during the competition
- Similar characteristics to public benchmarks
- Used to prevent overfitting
- Revealed after competition concludes

### Final Evaluation

1. **Baseline Evaluation**: Run Circuit Training on all benchmarks to establish baseline
2. **Submission Evaluation**: Run each submission on all benchmarks (public + hidden)
3. **Validation**: Check all placements for legality
4. **Scoring**: Compute aggregate improvement over baseline
5. **Ranking**: Sort by aggregate score (highest to lowest)
6. **Verification**: Verify winner's code and reproduce results

---

## Winner Selection

### Selection Criteria

The winner is the submission with the **highest aggregate score** that:

1. ✅ Achieves aggregate score > 0 (beats Circuit Training)
2. ✅ Produces valid placements on all benchmarks
3. ✅ Completes within timeout on all benchmarks
4. ✅ Passes reproducibility verification
5. ✅ Complies with all rules

### Verification Process

Before awarding the prize, we will:

1. **Code Review**: Examine winner's code for rule compliance
2. **Reproducibility Test**: Re-run winner's submission to verify results
3. **Interview** (optional): Discuss approach with winner (for fairness checking)

### Disqualification

Submissions may be disqualified for:

- Violating any competition rules
- Cheating or plagiarism
- Exploiting bugs in evaluation code
- Providing false information
- Failing reproducibility tests

### If No Winner

If **no submission achieves aggregate score > 0**:

- **No prize will be awarded**
- Results will be published
- Competition may be extended or relaunched with adjusted parameters

### Ties

In case of a tie (same aggregate score to 4 decimal places):

1. **Tiebreaker 1**: Highest minimum benchmark improvement (best worst-case performance)
2. **Tiebreaker 2**: Lowest total runtime across all benchmarks
3. **Tiebreaker 3**: First submission timestamp

If still tied, prize will be split equally.

---

## Terms and Conditions

### Prize Payment

- Prize will be paid within 30 days of winner verification
- Payment via bank transfer or PayPal (winner's choice)
- Winner responsible for any taxes on prize money
- Winner must provide tax information (W-9 or W-8BEN form)

### Intellectual Property

- **You retain all rights** to your submission code
- By submitting, you grant Partcl a non-exclusive license to:
  - Evaluate your submission
  - Publish aggregate results and rankings
  - Use for research and educational purposes (with attribution)
- Winner's code may be published (with consent) after competition

### Code of Conduct

Participants must:

- Be respectful to organizers and other participants
- Report bugs or issues promptly
- Not intentionally disrupt the competition
- Follow the [GitHub Community Guidelines](https://docs.github.com/en/site-policy/github-terms/github-community-guidelines)

### Liability

- Organizers are not liable for:
  - Technical issues beyond our control
  - Loss or corruption of submissions
  - Disputes between team members
- Organizers reserve the right to:
  - Modify rules with advance notice
  - Disqualify submissions for rule violations
  - Cancel or extend the competition if necessary

### Privacy

- Submission code will be kept confidential
- Results (aggregate scores, rankings) will be public
- Personal information will not be shared without consent

### Questions and Clarifications

- Questions about rules: Open an issue on GitHub
- Ambiguities will be clarified publicly (to maintain fairness)
- Rule changes will be announced with at least 2 weeks notice

---

## Summary of Key Points

🎯 **Goal**: Beat Circuit Training baseline on aggregate

💰 **Prize**: $20,000 USD (only if you beat the baseline)

📊 **Scoring**: Geometric mean of improvements across all benchmarks

⏱️ **Timeout**: 1 hour per benchmark

✅ **Constraints**: No overlaps, within boundaries, respect blockages

🚫 **Not Allowed**: Hardcoding, external APIs, cheating

🏆 **Winner**: Highest aggregate score > 0, with valid placements

---

## Acceptance of Rules

By submitting to this competition, you acknowledge that you have read, understood, and agree to abide by these rules and terms.

**Good luck, and may the best placer win!** 🚀

---

For questions or clarifications, please open an issue on [GitHub](https://github.com/partcl/partcl-marco-place-challenge/issues).
