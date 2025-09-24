import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def load_data(file_path, satellite):
    """Load dataset based on satellite type."""
    logger.info(f"Loading dataset for {satellite} from {file_path}")
    
    if satellite == "K2":
        try:
            df = pd.read_csv(file_path, comment="#")
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    else:
        logger.warning(f"Data for {satellite} is not available, skipping...")
        return None
