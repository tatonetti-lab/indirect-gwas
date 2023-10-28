#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "utils.hpp"

struct FeatureGwasResults
{
    std::vector<std::string> variant_ids;
    Eigen::VectorXd beta;
    Eigen::VectorXd std_error;
    Eigen::VectorXd degrees_of_freedom;

    FeatureGwasResults(unsigned int length)
    {
        variant_ids = std::vector<std::string>(length);
        beta = Eigen::VectorXd(length);
        std_error = Eigen::VectorXd(length);
        degrees_of_freedom = Eigen::VectorXd(length);
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
