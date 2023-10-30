#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "utils.hpp"

struct FeatureGwasResults
{
    std::vector<std::string> variant_ids;
    Eigen::VectorXf beta;
    Eigen::VectorXf std_error;
    Eigen::VectorXf degrees_of_freedom;

    FeatureGwasResults(unsigned int length)
    {
        variant_ids = std::vector<std::string>(length);
        beta = Eigen::VectorXf(length);
        std_error = Eigen::VectorXf(length);
        degrees_of_freedom = Eigen::VectorXf(length);
    };
};

unsigned int count_lines(std::string filename);

void head(std::string filename, unsigned int n_lines);

LabeledMatrix read_input_matrix(std::string filename);

FeatureGwasResults read_gwas_chunk(
    std::string filename,
    ColumnSpec column_names,
    unsigned int start_line,
    unsigned int end_line,
    unsigned int n_covariates);
