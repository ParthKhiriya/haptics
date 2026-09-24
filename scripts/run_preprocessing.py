import sys
from pathlib import Path

# Automatically inject the workspace root into sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import LMTPreprocessor

def main():
    # Pointing exactly to where your MATLAB files start
    raw_dir = PROJECT_ROOT / "data" / "LMT_FinalDatabase" / "CBM_FinalDatabase"
    
    # Keeping processed data neatly separated
    processed_dir = PROJECT_ROOT / "data" / "processed"

    print("========================================")
    print(" LMT Haptic Preprocessing Pipeline")
    print(f" Raw Directory:       {raw_dir}")
    print(f" Processed Directory: {processed_dir}")
    print("========================================")

    preprocessor = LMTPreprocessor(
        raw_dir=raw_dir,
        processed_dir=processed_dir
    )
    preprocessor.process()

if __name__ == "__main__":
    main()