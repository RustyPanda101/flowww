# ──────────────────────────────────────────────────────────────
# Configuration — extracted from optical_flow.ipynb Cells 4, 10, 43, 50
# ──────────────────────────────────────────────────────────────

# Cell 4 — Dataset location
DATASET_LOCATION = "FlyingChairs_release"

# Cell 10 — Dataset
DATA_ROOT   = DATASET_LOCATION
SUBSET_SIZE = 22872                                     # number of images you want to train on --- in this case i have set it to entire dataset size
                                                        # can be automated but later ... maybe 
VAL_SPLIT   = 0.1                                       # validation split ratio --- should probably incrase to 0.3 (70-30)
RESOLUTION  = 256                                       # resolution of the images essentially controls the spatial detail vs mem cost 

# Training
BATCH_SIZE    = 32                                      # training batch size --- greater the value more vram needed
EPOCHS        = 60                                      # do i really have to explain this? (number of epochs to train the model)
LOG_INTERVAL  = 100                                     # steps between logging values in output ... does not affect training at all

# Architecture
BASE_CH      = 96                                       # Base number of feature channels in encoder
MAX_DISP     = 4                                        # cost volume search radius: (2*4+1)**2 = 81 channels (standard RAFT value)
N_GRU_ITERS  = 6                                        # number of itterations for the convGRU --- higher value makes the inference slower but better refinement

# Optimisation
LR            = 1e-4                                    # learning rate cold start value
WEIGHT_DECAY  = 1e-6                                    # L2 regularization on weights
SCALE_WEIGHTS = [1.00, 0.50, 0.25]                      # Loss weights at different scales. Higher weight on coarse helps stabilize early training.
SMOOTHNESS_W  = 0.0001                                  # Regularizes flow field to be smooth, basically a control for how sharp the edges should be (should probably decrease it)
GRAD_W        = 0.02                                    # gradient loss weight

# Cell 43 — Photometric loss weight
PHOTO_W = 0.25                                          # photometric loss weight

# Cell 50 — Final model name
final_optic_flow_model_name = "heh"
