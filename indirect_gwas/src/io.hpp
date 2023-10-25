#pragma once

#include <string>

#include "utils.hpp"

LabeledMatrix read_input_matrix(std::string filename);
unsigned int count_lines(std::string filename);
void head(std::string filename, unsigned int n_lines);
