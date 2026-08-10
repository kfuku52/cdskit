#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdskit.targetp_external_aug import main


if __name__ == "__main__":
    main()
