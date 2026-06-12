#include "sampler.h"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

// Turn scores into probabilities that sum to 1.
// Same "subtract the max first" stability trick as the Python version.
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

// The deterministic math half: scores in, probabilities out.
// For now only temperature. We add top-k and top-p next.
std::vector<double> sample_logits(std::vector<double> logits,
                                  double temperature) {
    // Greedy special case: temperature 0 means always the highest score.
    if (temperature == 0.0) {
        std::vector<double> probs(logits.size(), 0.0);
        size_t best = 0;
        for (size_t i = 1; i < logits.size(); ++i) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        probs[best] = 1.0;
        return probs;
    }

    // Temperature: divide every score.
    for (double& v : logits) {
        v /= temperature;
    }

    return softmax(logits);
}