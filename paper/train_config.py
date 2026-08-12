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

# TODO(B): PIN THIS BEFORE STARTING P0.
#
# Left to B because B is the one who downloads the weights, and the revision
# that actually gets used is only knowable on that machine. After the first
# download, read the resolved commit hash from either
#   ~/.cache/huggingface/hub/models--hfl--chinese-roberta-wwm-ext-large/snapshots/<hash>/
# or `model.config._commit_hash`, then hard-code it here so every one of the 15
# fits requests that exact revision rather than whatever the Hub serves that day.
#
# Leaving it None is caught at launch: paper/run_training.py refuses to start
# unless it is pinned, so no GPU time is spent on weights that cannot be
# identified afterwards.
MODEL_REVISION = None

MAX_LEN = 384

# --- Optimisation ---------------------------------------------------------
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2          # effective batch = 16

# Fixed budget: every rotation trains exactly this many epochs and there is no
# early stopping (see the checkpoint block below for why).
#
# TODO(B): SET THIS FROM THE COMPETITION TRAINING LOGS BEFORE STARTING P0.
#
# 12 is a placeholder inherited from the competition configuration, where early
# stopping with patience=3 backstopped it -- runs there rarely reached epoch 12,
# so the number itself was never validated. The logs are not in this repository
# (the competition .gitignore excluded *.log); they should still be on the GPU
# machine.
#
#   What to look for: the epoch at which early stopping typically fired across
#   the competition folds -- that is the only empirical evidence available.
#   What to set: that epoch, rounded up. Record the chosen value and the log
#   evidence in the run manifest.
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
# This constant is the one deliberate exception to the protocol freeze: it is
# left open for B to set, and is frozen from the moment P0 starts. Changing it
# afterwards invalidates every fit completed before the change.
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
