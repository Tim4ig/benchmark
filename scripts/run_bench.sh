#!/usr/bin/env bash
# run_bench.sh - Build and run all backends, collect CSV, validate it, and generate plots.
#
# Usage:
#   ./scripts/run_bench.sh [build_dir] [results_dir]
#
# Defaults:
#   build_dir   = ./build
#   results_dir = ./results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${1:-${PROJECT_DIR}/build}"
RESULTS_DIR="${2:-${PROJECT_DIR}/results}"

mkdir -p "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# System performance setup - remove all power/frequency limits
# ---------------------------------------------------------------------------
echo "=== System performance setup ==="

# CPU: ensure performance governor and boost are enabled on all cores
if command -v cpupower &>/dev/null; then
  sudo cpupower frequency-set -g performance 2>/dev/null && echo "  CPU governor: performance" || true
else
  for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee "$gov" > /dev/null 2>&1 || true
  done
  echo "  CPU governor: performance (set via sysfs)"
fi

CPU_BOOST="/sys/devices/system/cpu/cpufreq/boost"
if [[ -f "$CPU_BOOST" ]]; then
  BOOST_VAL=$(cat "$CPU_BOOST")
  if [[ "$BOOST_VAL" != "1" ]]; then
    echo 1 | sudo tee "$CPU_BOOST" > /dev/null && echo "  CPU boost: enabled"
  else
    echo "  CPU boost: already enabled"
  fi
fi

# CPU: disable RAPL power limits (EPP = performance)
for epp in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do
  echo performance | sudo tee "$epp" > /dev/null 2>&1 || true
done
echo "  CPU EPP: performance"

# CPU RAPL: remove power caps if any are set
RAPL_ROOT="/sys/class/powercap/intel-rapl:0"
if [[ -f "${RAPL_ROOT}/constraint_0_power_limit_uw" ]]; then
  CAP_MAX=$(cat "${RAPL_ROOT}/constraint_0_max_power_uw" 2>/dev/null || echo "")
  if [[ -n "$CAP_MAX" ]]; then
    echo "$CAP_MAX" | sudo tee "${RAPL_ROOT}/constraint_0_power_limit_uw" > /dev/null 2>&1 \
      && echo "  CPU RAPL cap: set to max (${CAP_MAX} µW = $((CAP_MAX / 1000000)) W)" \
      || echo "  CPU RAPL cap: could not set (not critical)"
  fi
fi

# GPU: set power cap to max
GPU_HWMON=$(grep -rl "^amdgpu$" /sys/class/hwmon/*/name 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
if [[ -n "$GPU_HWMON" && -f "$GPU_HWMON/power1_cap" ]]; then
  GPU_CAP=$(cat "$GPU_HWMON/power1_cap")
  GPU_CAP_MAX=$(cat "$GPU_HWMON/power1_cap_max")
  if [[ "$GPU_CAP" -lt "$GPU_CAP_MAX" ]]; then
    echo "$GPU_CAP_MAX" | sudo tee "$GPU_HWMON/power1_cap" > /dev/null \
      && echo "  GPU power cap: raised to max ($((GPU_CAP_MAX / 1000000)) W)"
  else
    echo "  GPU power cap: already at max ($((GPU_CAP / 1000000)) W)"
  fi
fi

# RAPL read permission for power measurement in orchestrator
echo "  RAPL read permission: granting..."
sudo chmod o+r /sys/class/powercap/intel-rapl*/energy_uj 2>/dev/null \
  && echo "  RAPL read permission: OK" \
  || echo "  RAPL read permission: failed (sudo may not be available; watts_cpu will be -1)"

echo ""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "=== Building in ${BUILD_DIR} ==="
cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -G "Unix Makefiles" 2>&1 | tail -5
cmake --build "${BUILD_DIR}" -j"$(nproc)" 2>&1 | tail -20
echo ""

# ---------------------------------------------------------------------------
# Locate libraries
# ---------------------------------------------------------------------------
ORCHESTRATOR="${BUILD_DIR}/orchestrator/bench_orchestrator"
LIB_CPU_REF="${BUILD_DIR}/cpu_ref/libcpu_ref.so"
LIB_CPU_AUTO="${BUILD_DIR}/cpu_auto/libcpu_auto.so"
LIB_CPU_AVX512="${BUILD_DIR}/cpu_avx512/libcpu_avx512.so"
LIB_VK="${BUILD_DIR}/vkbench/libvkbench.so"
LIB_CPU_MT="${BUILD_DIR}/cpu_mt/libcpu_mt.so"
LIB_HYBRID="${BUILD_DIR}/hybrid/libhybrid.so"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CSV="${RESULTS_DIR}/bench_${TIMESTAMP}.csv"
LATEST_CSV="${RESULTS_DIR}/latest.csv"

LIBS=()
for L in "$LIB_CPU_REF" "$LIB_CPU_AUTO" "$LIB_CPU_AVX512" "$LIB_CPU_MT" "$LIB_VK" "$LIB_HYBRID"; do
  if [[ -f "$L" ]]; then
    LIBS+=("$L")
    echo "  Found: $(basename $L)"
  else
    echo "  MISSING: $(basename "$L") - skipping"
  fi
done

if [[ ${#LIBS[@]} -eq 0 ]]; then
  echo "ERROR: No libraries found. Did the build succeed?"
  exit 1
fi

# ---------------------------------------------------------------------------
# Run benchmarks (20 repeats default via kDefaultRepeats in bench_settings.h)
# ---------------------------------------------------------------------------
echo ""
echo "=== Running benchmarks (repeats=20) -> ${CSV} ==="
"${ORCHESTRATOR}" --csv "${CSV}" "${LIBS[@]}" 2>&1 | tee "${RESULTS_DIR}/bench_${TIMESTAMP}.log"

cp -f "${CSV}" "${LATEST_CSV}"
echo ""
echo "CSV saved: ${CSV}"
echo "Latest:    ${LATEST_CSV}"

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if command -v python3 &>/dev/null; then
  echo ""
  echo "=== Validating CSV ==="
  python3 "${SCRIPT_DIR}/validate_results.py" "${LATEST_CSV}"
else
  echo "python3 not found - skipping validation"
fi

# ---------------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------------
PLOTS_DIR="${RESULTS_DIR}/plots_${TIMESTAMP}"
echo ""
echo "=== Generating plots -> ${PLOTS_DIR} ==="
if command -v python3 &>/dev/null; then
  python3 "${SCRIPT_DIR}/plot_results.py" "${LATEST_CSV}" "${PLOTS_DIR}" || true
else
  echo "python3 not found - skipping plots"
fi

echo ""
echo "=== Done ==="
echo "Results in: ${RESULTS_DIR}"
