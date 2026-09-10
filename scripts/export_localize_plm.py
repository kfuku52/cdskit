#!/usr/bin/env python3
"""Export a frozen-PLM checkpoint without training-machine cache restrictions."""

import argparse
from copy import deepcopy
from pathlib import Path
import re

from cdskit.localize_model import load_localize_model, save_localize_model
from cdskit.localize_multilabel_plm import ResidueEncoder


def portable_model(model):
    """Preserve learned parameters and encoder identity; reject local encoders."""
    if model.get("model_type") != "multilabel_plm_v1":
        raise ValueError("Export requires a multilabel_plm_v1 checkpoint.")
    result = deepcopy(model)
    head = result["localization_model"]
    encoder = head["encoder"]
    name = str(encoder.get("model_name", ""))
    if (
        not re.fullmatch(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*", name)
        or Path(name).exists()
    ):
        raise ValueError("Portable export requires a remote owner/model encoder ID.")
    if ResidueEncoder(encoder).identity != head["encoder_identity"]:
        raise ValueError("Encoder identity differs from the trained checkpoint.")
    encoder["cache_dir"] = ""
    encoder["local_files_only"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    if Path(args.destination).exists():
        parser.error("Destination exists; choose a new export path.")
    save_localize_model(
        portable_model(load_localize_model(args.source)), args.destination
    )


if __name__ == "__main__":
    main()
