#pragma once
#include <vector>

std::vector<double> sample_logits(std::vector<double> logits,
                                  double temperature,
                                  int top_k,
                                  double top_p);