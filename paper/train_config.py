"""Frozen training recipe for the fixed base-model anchor.

There are deliberately no ablation switches here. The protocol fixes every
official run to one standard-loss recipe, and a constant that does not exist
cannot be flipped between runs.

Nothing here may change after the protocol freeze gate without A's sign-off
and a manifest entry.
"""

# --- Backbone -------------------------------------------------------------
# Frozen anchor. Despite the name, HFL's Chinese RoBERTa-wwm-ext is a BERT
# *architecture* trained RoBERTa-style (whole word masking, more data, no NSP)
# and loads through BertModel/BertTokenizer -- the Methods section must say so
# rather than implying the RoBERTa architecture.
MODEL_NAME = "hfl/chinese-roberta-wwm-ext-large"

# Resolved on B's GPU machine from both the Hugging Face cache snapshot path
# and refs/main before the first official run (2026-08-15). Every fit requests
# this exact commit rather than whatever the Hub serves later.
MODEL_REVISION = "a25cc9e05974bd9687e528edd516f2cfdb3f5db9"

MAX_LEN = 384

# --- Optimisation ---------------------------------------------------------
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2          # effective batch = 16

# Fixed budget: every rotation trains exactly this many epochs and there is no
# early stopping (see the checkpoint block below for why).
#
# B re-audited the original standard-loss Chinese RoBERTa logs on 2026-08-21.
# Those runs used a 15-epoch cap with patience=3. Reading "Early stop at epoch"
# (or 15 for the one fold that exhausted the cap), their terminal epochs were:
#
#   seed 42:  12, 10, 10, 15, 9
#   seed 123: 14, 14, 10,  6, 14
#   seed 456: 12, 11, 11, 14, 9
#
# Define the protocol's formerly ambiguous "typical stopping epoch, rounded
# up" as the ceiling of the arithmetic mean across all 15 folds:
# ceil(171 / 15) = ceil(11.4) = 12. The median is 11, but is descriptive only;
# it is not the frozen aggregation rule. The per-fold audit and hashes of the
# three source logs are recorded in docs/competition_epoch_evidence.md.
#
# Why it matters in both directions, and why nobody downstream can catch a bad
# value:
#   * too high overfits, too low undertrains, and neither is detectable -- the
#     protocol leaves no labelled split to check a checkpoint against;
#   * without early stopping every one of the 15 fits now runs the full budget,
#     so each surplus epoch costs GPU wall-clock on the critical path. If the
#     logs show runs settling around epoch 8, keeping 12 burns ~50% more GPU
#     time for nothing.
#
# Frozen before P0. Changing it invalidates every completed official fit.
EPOCHS = 12

BACKBONE_LR = 2e-5
HEAD_LR = 1e-4

# Stated here rather than left to torch's AdamW default, which is also 0.01
# today. A hyperparameter that lives in a library default is not frozen: it can
# move with a version bump, and train_config_sha256 -- the hash every bundle
# records as its proof of recipe -- would not change when it did.
WEIGHT_DECAY = 0.01

# Biases and LayerNorm gains are exempt, the standard BERT fine-tuning
# convention: decaying a normalisation gain toward zero fights the layer's
# purpose. Matched by substring against parameter names.
NO_DECAY = ("bias", "LayerNorm.weight")

# AdamW's betas (0.9, 0.999) and eps (1e-8) stay at the torch defaults. They
# have not moved across releases and no recipe here depends on them; if that
# ever stops being true they belong here too.

WARMUP_RATIO = 0.1
LR_SCHEDULE = "cosine"        # "linear" | "cosine"
DROPOUT = 0.1
LABEL_SMOOTHING = 0.1
USE_MEAN_POOLING = True
USE_LLRD = True               # layer-wise learning rate decay
LLRD_DECAY = 0.9
MAX_GRAD_NORM = 1.0

# Class weighting
MAX_CLASS_WEIGHT = 10.0
MIN_CLASS_SAMPLES_FOR_UPWEIGHT = 5

# --- Checkpoint selection -------------------------------------------------
# DECIDED: fixed epoch budget, average the last K checkpoints. No evaluation
# label is consulted at any point.
#
# The protocol forbids using the Calibration or Test partition for early
# stopping, and the rotating design leaves no other labelled split, so
# label-driven checkpoint selection is not available at all. Selecting on Test
# would inflate the reported score outright; selecting on Calibration would
# make that partition do two jobs at once, entangling model choice with the
# class-bias fit that M2/M3/M5/M6 are supposed to isolate.
#
# Rejected alternative: carve an inner validation slice out of Train. Legal,
# but a 10% slice of ~1,200 training rows holds almost no rare-class support,
# so the early-stopping signal would be mostly noise, and it adds a nuisance
# parameter.
CHECKPOINT_RULE = "avg_last_k"

# Average the LAST K states, not the best K. "Best" would require scoring
# checkpoints against labels this protocol puts out of reach; K itself is
# carried over from the competition's top-3 averaging.
CHECKPOINT_LAST_K = 3

# --- Protocol -------------------------------------------------------------
SEEDS = (42, 123, 456)
N_FOLDS = 5
PROTOCOLS = ("pdf_group", "row_strat")
