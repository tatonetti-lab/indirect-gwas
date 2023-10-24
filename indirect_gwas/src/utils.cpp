#include "utils.hpp"
#include <iostream>

double compute_log_p_value(double t_statistic, unsigned int degrees_of_freedom) {
    boost::math::students_t dist(degrees_of_freedom);
    double p_value = boost::math::cdf(dist, -std::abs(t_statistic));
    return -1. * (std::log(p_value) + std::log(2)) / std::log(10);
}
