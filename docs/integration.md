# Integration Guide

## Choose an Adapter

For a standard MLP with a literal final linear layer, start with
`LinearLastLayerAdapter`. A rank-2 model output gains a singleton horizon axis;
targets must explicitly gain the same axis. Rank-3 outputs already use [B,H,C].
Scalar regression still needs C=1. Classification/logits are outside this API.

For custom models, return `Forecast` from a callable. Compute mean and features
together rather than invoking an expensive trunk twice. SPARC calls this callable
once per prediction; it cannot ensure the callable itself does only one forward.
Use evaluation mode and fixed preprocessing, including normalization and units.
Supply features on the mean device, and include the design's bias coordinate
explicitly. Feature scale changes the ridge prior, so do not normalize at test time
without refitting and recalibrating.

For multi-horizon heads flattened as [B,H*C], the generic linear adapter treats
this as H=1 with HC coordinates. Use a custom adapter to reshape the mean and
construct correctly aligned per-horizon design vectors if horizon-wise kappa is
desired. Never reshape features arbitrarily to make dimensions fit.

For non-neural predictors, use an explicitly chosen, fixed representation. Return
PyTorch tensors from the callable. This is a leverage-based plug-in score, not an
exact last-layer posterior unless the assumed linear model really applies.

## Use Four Logical Stages

1. Train/select the point predictor on training/validation data.
2. Freeze it; fit feature statistics and any residual covariance on a model-fit
   split. A separate residual-fit split avoids optimistic in-sample residuals.
   Pretrained feature parameters themselves may have used training data.
3. Calibrate once on held-out, exchangeable units with all choices frozen.
4. Evaluate on untouched test units. Report coverage AND width, not coverage alone.

No loader may silently mix subjects/sequences across splits. Move input batches to
the model device yourself; the package supports arbitrary input objects and cannot
recursively infer how to move dictionaries or model-specific objects. Targets may
remain on CPU; fitting and calibration accumulate float64 CPU statistics.

## Memory and Performance

Feature fitting needs O(H*P^2) state and processes batches incrementally. Prediction
needs triangular solves of approximately O(B*H*P^2), in addition to one predictor
call. This is not free for very wide features. Explicit low-dimensional design
choices should be fitted before calibration and evaluated for utility.

Calibration currently stores O(N*H*C) scores for exact quantiles. The optional dense
covariance factor uses O(B*(H*C)^2) memory; it is intended for small examples and
tests. Supplied-factor validation also checks nonsingularity and is cubic in HC.
Do not present this initial dense path as the paper's optimized structured runtime.
Large-body forecasting needs a subsequent factored-operator adapter.

## State and Maintenance

Checkpoint model weights and preprocessing separately. Use a model_id containing
their version identifiers; it is not auto-validated against the actual weights.
Changing mean, feature map, covariance head, units or output order requires
refitting/recalibration. Use `predict(intervals=False)` explicitly when only raw
model uncertainty is needed before calibration.

Choose alpha and scope before calibration. Public mutation of fitted internals is
unsupported. The calibration set may be replaced by a new valid held-out set via
`calibrate`; an empty/failed refit leaves the previous valid state intact.
