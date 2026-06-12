#include "sampler.h"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

static std::vector<double> softmax(const std::vector<double>& scores) {
    double max_score = *std::max_element(scores.begin(), scores.end());
    std::vector<double> result(scores.size());
    double sum = 0.0;
    for (size_t i = 0; i < scores.size(); ++i) {
        result[i] = std::exp(scores[i] - max_score);
        sum += result[i];
    }
    for (double& v : result) {
        v /= sum;
    }
    return result;
}

std::vector<double> sample_logits(std::vector<double> logits,
                                  double temperature,
                                  int top_k,
                                  double top_p) {
    const size_t n = logits.size();

    // Greedy: temperature 0 means always the single best token.
    if (temperature == 0.0) {
        std::vector<double> probs(n, 0.0);
        size_t best = 0;
        for (size_t i = 1; i < n; ++i) {
            if (logits[i] > logits[best]) best = i;
        }
        probs[best] = 1.0;
        return probs;
    }

    // STEP 1 - temperature
    for (double& v : logits) {
        v /= temperature;
    }

    // STEP 2 - top-k: keep only the k highest scores.
    if (top_k > 0 && static_cast<size_t>(top_k) < n) {
        // Copy scores, find the k-th largest value, mask everything below it.
        std::vector<double> sorted = logits;
        std::nth_element(sorted.begin(),
                         sorted.begin() + (n - top_k),
                         sorted.end());
        double kth_best = sorted[n - top_k];
        for (double& v : logits) {
            if (v < kth_best) v = -std::numeric_limits<double>::infinity();
        }
    }

    // STEP 3 - probabilities
    std::vector<double> probs = softmax(logits);

    // STEP 4 - top-p: keep the smallest set of top tokens summing to >= p.
    if (top_p < 1.0) {
        // order = token indices sorted by probability, highest first
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) { return probs[a] > probs[b]; });

        double running = 0.0;
        size_t keep_count = 0;
        for (size_t i = 0; i < n; ++i) {
            running += probs[order[i]];
            keep_count = i + 1;
            if (running >= top_p) break;
        }

        // zero out everything past the kept set
        std::vector<double> trimmed(n, 0.0);
        double kept_sum = 0.0;
        for (size_t i = 0; i < keep_count; ++i) {
            trimmed[order[i]] = probs[order[i]];
            kept_sum += probs[order[i]];
        }
        for (double& v : trimmed) {
            v /= kept_sum;  // renormalize so survivors sum to 1
        }
        probs = trimmed;
    }

    return probs;
}