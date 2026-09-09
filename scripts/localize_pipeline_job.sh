#!/usr/bin/env bash
#SBATCH --job-name=cdskit-localize
#SBATCH --output=cdskit-localize-%j.log
# Supply site-approved partition/GPU/CPU/memory/time options to sbatch.
# Install this checkout in the selected environment before submitting.
set -euo pipefail
exec "${CDSKIT_PYTHON:-python3}" -m cdskit.cli localize-learn "$@"
