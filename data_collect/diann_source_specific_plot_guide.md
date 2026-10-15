# Plot guide: source-specific DIA-NN validation sweep

All figures are **validation-only**. Every performance value is observed RT MAE in minutes after the target-specific inverse mapping. The final test is still unopened, so the observations below guide interpretation and finalist selection but are not final performance claims. Target coordinates are faceted rather than pooled.

## fig1 — Overall versus Heavy-Dimethyl Pareto screen

**Thesis rating: 70/100 — strong candidate for main text or supplement.**

**What it shows.** Every point is one strategy averaged over three seeds. The x-axis is source-balanced overall MAE; the y-axis is source-balanced Heavy-Macro MAE. Circles and diamonds mark the validation-selected overall and Heavy finalists.

**What is interesting here.** Source-gradient normalized RT has a much larger cross-source error range than the aligned coordinates. Its good in-domain fit does not translate into a good all-source model.

## fig2 — Validation-selected candidate uncertainty

**Thesis rating: 20/100 — keep as internal selection/QC figure; omit from thesis main text.**

**What it shows.** Compact ranking of the two finalists per training scope. Error bars are 95% normal-approximation intervals over three optimization seeds, not biological-replicate intervals.

**What is interesting here.** Many overall and Heavy finalists coincide or are very close. There is no universally dominant Heavy-specific recipe.

## fig3 — In-domain fit versus cross-source transfer

**Thesis rating: 78/100 — strong main-text candidate; concise visual statement of the specialization cost.**

**What it shows.** Each source-specific finalist is evaluated on its own source (x-axis) and averaged over all four evaluation sources (y-axis). The dashed diagonal denotes equal in-domain and all-source error.

**What is interesting here.** Source-gradient normalized models lie far above the diagonal: they fit their source well but transfer poorly. Chronologer HI and Human-anchor models are substantially closer to the diagonal.

## fig4 — Heavy-Dimethyl strata

**Thesis rating: 62/100 — useful if Heavy-Dimethyl performance is a central thesis result; otherwise supplement.**

**What it shows.** Selected models are separated into N-terminal-only, lysine-only and doubly-heavy peptide strata, so Heavy-Macro cannot hide a weak stratum.

**What is interesting here.** Doubly-heavy peptides are consistently most difficult. Source-gradient normalized RT inflates all three strata outside the source domain, not only the doubly-heavy subset.

## fig5 — Coordinate robustness by source

**Thesis rating: 55/100 — useful methodological support; prefer supplement unless normalization is a main thesis chapter.**

**What it shows.** One panel per acquisition source, comparing in-domain overall finalists across RT coordinates with seed uncertainty. The Human anchor point is not applicable because it is identical to Human source-gradient normalization.

**What is interesting here.** In-domain MAEs are very similar across target coordinates. The practical difference is transferability, rather than a large in-domain accuracy gain.

## fig6 — Directed transfer matrices

**Thesis rating: 88/100 — main-text figure. It makes the key transfer pattern immediately visible.**

**What it shows.** Rows are source-specific training datasets and columns are evaluation datasets. Diagonal cells are in-domain; off-diagonal cells quantify directional transfer.

**What is interesting here.** Source-gradient normalized RT has very large off-diagonal errors. Chronologer HI and Human-anchor remain much lower off the diagonal, although C. elegans is still a difficult transfer domain.

## fig7 — Effects of all strategies relative to baseline

**Thesis rating: 45/100 — good evidence that the broad sweep was informative; use in supplement or methods results.**

**What it shows.** Every point is a strategy relative to that source's own unweighted MAE baseline. Left/down is better: lower overall and lower Heavy-Macro error. Color denotes strategy family; marker denotes source.

**What is interesting here.** Most aggressive weighting/oversampling settings move rightwards or upwards. Improvements are local, not a consistent sweep-wide Heavy-Dimethyl effect; simple MAE/LR choices remain competitive.

## fig8 — Optimization-seed stability

**Thesis rating: 15/100 — retain for QC/supplement only; it supports robustness but is not part of the biological or modeling story.**

**What it shows.** For every source and all 42 strategies, this is the standard deviation of in-domain MAE over the three fixed seeds.

**What is interesting here.** Typical seed variation is only a few thousandths of a minute. Larger spread in some Human and source-gradient-normalized settings warns against overreading very small ranking differences.

## fig9 — Realised in-source Overall/Heavy trade-off

**Thesis rating: 50/100 — compact support for the Heavy-selection decision; supplement by default.**

**What it shows.** A line connects the overall and Heavy finalist for every training source. This is its intended in-source use, not a cross-source average.

**What is interesting here.** Most pairs are tightly clustered. Heavy-specific selection makes only modest changes, consistent with fig7.

## fig10 — Source-specific versus matched combined reference

**Thesis rating: 82/100 — strongest current validation comparison. Replace it with the identical final-test figure; that replacement is a 100/100 main-text figure if the pattern replicates.**

**What it shows.** Each cell is source-specific MAE minus a target-matched combined reference on the same validation source. Negative favors the source-specific model. The source-gradient panel uses the new combined model; Chronologer and Anchor use frozen historical combined models with the exact same split.

**What is interesting here.** C. elegans and both TripleMix sources improve on their own source, while Human gains little. The same models often worsen other sources, supporting source-specific expert models rather than replacement of a combined generalist. This must be confirmed once on the final test.
