#!/usr/bin/env python3
"""Prepare datasets for AEQ-DETR."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_prep.prepare import main
if __name__ == '__main__':
    main()
