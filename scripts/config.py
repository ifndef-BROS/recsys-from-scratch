from pathlib import Path


# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = DATA_DIR / "models"

ALL_DIRS = {DATA_DIR, EMBEDDINGS_DIR, MODELS_DIR}

# Dataset Settings
CATEGORY = "Software"
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5
# Upper limit for max number of interactions considered for user embedding
MAX_HISTORY_LENGTH = 50

# For Embedding Generation
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384     # Output from miniLM
REDUCED_DIM = 128       # After PCA

# For Evaluation
TOP_K = [5, 10, 20]
TEST_SPLIT = "leave_one_out"


# For testing
RANDOM_SEED = 11


if __name__ == "__main__":
    for directory in ALL_DIRS:
        directory.mkdir(exist_ok=True)