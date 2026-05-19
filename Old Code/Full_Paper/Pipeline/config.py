# ==========================================================
# PROTOCOL - GENERAL
# ==========================================================

SAMPLE_RATE = 48000
BEACON_FREQ = 10000
BEACON_DURATION_SEC = 2.0

# ==========================================================
# PROTOCOL GENERATOR
# ==========================================================

# --- Alignment structure ---
INITIAL_SILENCE_SEC = 15.0
GUARD_SILENCE_SEC   = 1.5
TAIL_SILENCE_SEC    = 5.0

# --- Chirp pulse train ---
PULSE_DURATION = 0.10
GAP_DURATION   = 0.05
START_FREQ     = 15000.0
END_FREQ       = 19200.0
AMPLITUDE      = 0.85
FADE_MS        = 5.0

CYCLES_TOTAL   = 2
ACTIVE_SECS    = 60.0

# ==========================================================
# PROTOCOL DECODER
# ==========================================================

BEACON_BANDWIDTH = 300            # ± Hz around beacon frequency
MIN_BEACON_DURATION = 2.5  # seconds
THRESHOLD_RATIO = 0.3