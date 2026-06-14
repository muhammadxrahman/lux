from prometheus_client import Counter, Gauge, Histogram

# Counters: only go up. Rates are computed from them later.
REQUESTS = Counter(
    "inference_requests_total",
    "Total inference requests handled",
    ["mode"],                       # label: "batch" or "stream"
)
TOKENS = Counter(
    "inference_tokens_generated_total",
    "Total tokens generated across all requests",
)
PREFIX_CACHE = Counter(
    "inference_prefix_cache_total",
    "Prefix-cache lookups, split by result",
    ["result"],                     # label: "hit" or "miss"
)

# Gauges: current snapshot, can rise and fall.
QUEUE_DEPTH = Gauge(
    "inference_queue_depth",
    "Jobs waiting in the scheduler queue",
)
INFLIGHT = Gauge(
    "inference_inflight_requests",
    "Requests currently generating",
)
ACTIVE_SEQUENCES = Gauge(
    "inference_active_sequences",
    "Sequences decoding together in the current step (continuous-batch width)",
)

# Histograms: distributions, so we can ask for percentiles later.
BATCH_SIZE = Histogram(
    "inference_batch_size",
    # With continuous batching there is no discrete batch; we observe the
    # number of sequences that decoded together in each step.
    "Sequences decoded together per step",
    buckets=[1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32],
)
GENERATION_SECONDS = Histogram(
    "inference_generation_seconds",
    "Wall-clock time per decode step",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2],
)