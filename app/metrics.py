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

# Histograms: distributions, so we can ask for percentiles later.
BATCH_SIZE = Histogram(
    "inference_batch_size",
    "How many requests were grouped into each batch",
    buckets=[1, 2, 3, 4, 5, 6, 7, 8, 12, 16],
)
GENERATION_SECONDS = Histogram(
    "inference_generation_seconds",
    "Wall-clock time per generation call",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30],
)