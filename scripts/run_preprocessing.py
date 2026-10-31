import sys
from pathlib import Path

# Automatically inject workspace root into sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess_sliding import LMTSlidingPreprocessor

def main():
    raw_dir = PROJECT_ROOT / "data" / "LMT_FinalDatabase" / "CBM_FinalDatabase"
    processed_dir = PROJECT_ROOT / "data" / "processed_sliding"

    print("========================================")
    print(" LMT Controlled Sliding Preprocessing Pipeline")
    print(f" Raw Directory:       {raw_dir}")
    print(f" Processed Directory: {processed_dir}")
    print("========================================")

    preprocessor = LMTSlidingPreprocessor(
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir)
    )
    preprocessor.process()

if __name__ == "__main__":
    main()
