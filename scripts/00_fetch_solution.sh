#!/usr/bin/env bash
# Fetch one daily Gualandi Cascadia solution.
#
#   ./scripts/00_fetch_solution.sh 2026-02-24 data/
#
# The six files below were confirmed present in the directory listing for
# 2026-02-24 (total ~96 MB, matching the "~100 MB per day" in Gualandi 2025).
# Directories exist for most days from 2024-07-08 onward; some days are
# missing, so if a date 404s pick the neighbouring one.
#
# You only need ONE dated directory. Each daily solution reconstructs the whole
# spatiotemporal history from the full GNSS record, so a recent date gives you
# 2007 to that date. Downloading many dates is only useful if you want to study
# how the near-real-time solution itself evolves.

set -euo pipefail

DATE="${1:?usage: $0 YYYY-MM-DD [outdir]}"
OUTDIR="${2:-data}/${DATE}"
BASE="https://near-real-time-sse.esc.cam.ac.uk/cascadia/${DATE}"

mkdir -p "${OUTDIR}"

# ICA.mat        ~25 MB  vbICA decomposition: U, S, V, var_U, var_V
# X.mat          ~38 MB  input position time series, station llh, timeline
# fitresult.mat  ~31 MB  trajectory-model fit (secular + seasonal + offsets)
# options.mat    ~1.8 MB inversion, smoothing and rate settings; fault.nu
# ind_comps.mat  ~2 KB   indices of the components retained
# misfit_comps.mat ~3 KB cross-validation misfit curves
for f in options.mat ind_comps.mat misfit_comps.mat ICA.mat X.mat fitresult.mat; do
  echo "--> ${f}"
  curl -fSL --retry 3 -o "${OUTDIR}/${f}" "${BASE}/${f}"
done

echo
echo "Downloaded to ${OUTDIR}:"
ls -lh "${OUTDIR}"
echo
echo "NOTE: the triangular fault mesh is NOT in this download."
echo "      options.fault.fault_file names it, but the file itself lives in"
echo "      Gualandi's own data tree (dir_data/Faults/<case>/). See"
echo "      docs/RUN_ON_REAL_DATA.md."
echo
echo "Next:  python scripts/01_inspect_solution.py ${OUTDIR}"
