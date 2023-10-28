#include "utils.hpp"

#include <cmath>
#include <ctime>
#include <iostream>
#include <string>

#include <boost/math/distributions/students_t.hpp>

float compute_log_p_value(float t_statistic, unsigned int degrees_of_freedom)
{
    boost::math::students_t dist(degrees_of_freedom);
    float p_value = boost::math::cdf(dist, -std::abs(t_statistic));
    return -1. * (std::log(p_value) + std::log(2)) / std::log(10);
}

// Produce a human readable current datetime string
std::string get_formatted_time()
{
    std::time_t now = std::time(nullptr);
    char buf[sizeof("YYYY-MM-DD HH:MM:SS")];
    std::strftime(buf, sizeof(buf), "%F %T", std::localtime(&now));
    return std::string(buf);
}
