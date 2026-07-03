from pathlib import Path
import sys


src_path = Path(__file__).resolve().parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from pyneuroscope.app import main


if __name__ == "__main__":
    main()
