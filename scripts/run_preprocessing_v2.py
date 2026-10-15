"""Preprocessing runner that backs up old data and runs the fixed pipeline."""
import sys
import shutil
from pathlib import Path

# Inject project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import LMTPreprocessor

def main():
    raw_dir = PROJECT_ROOT / "data" / "LMT_FinalDatabase" / "CBM_FinalDatabase"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    backup_dir = PROJECT_ROOT / "data" / "processed_old"
    
    # Backup old processed data
    if processed_dir.exists():
        print(f"Backing up {processed_dir} -> {backup_dir}")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.move(str(processed_dir), str(backup_dir))
    
    # Run preprocessing
    preprocessor = LMTPreprocessor(str(raw_dir), str(processed_dir))
    preprocessor.process()

if __name__ == "__main__":
    main()