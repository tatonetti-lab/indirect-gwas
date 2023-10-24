#pragma once

#include <algorithm>
#include <unordered_map>
#include "io.hpp"

class IndirectGWAS
{
private:
    // Names of columns in the input files
    ColumnSpec column_names;

    // Parameters
    const unsigned int n_covariates;
    const unsigned int chunksize;

    // Input data
    LabeledMatrix projection_coefficients;
    LabeledVector feature_partial_variance;

    // Empty vectors to hold data from each chunk
    std::vector<std::string> variant_ids;
    Eigen::VectorXd beta_vec;
    Eigen::VectorXd std_error_vec;
    Eigen::VectorXd sample_size_vec;

    // Computed data
    LabeledVector projection_partial_variance;
    LabeledVector degrees_of_freedom;
    LabeledMatrix beta;
    LabeledVector gpv_sum;
    unsigned int n_features;
    unsigned int n_projections;

    // Functions
    std::unordered_map<std::string, unsigned int> get_header_indexes(std::string header_line);
    void read_file_chunk(std::string filename, unsigned int start_row, unsigned int end_row);
    void process_file_chunk(unsigned int k, std::string filename, unsigned int start_row, unsigned int end_row);
    ResultsChunk compute_results_chunk();
    void save_results_chunk(ResultsChunk &results, std::string output_stem, bool write_header);

public:
    IndirectGWAS(
        ColumnSpec column_names,
        std::string projection_coefficients_filename,
        std::string feature_partial_covariance_filename,
        const unsigned int n_covariates,
        const unsigned int chunksize);
    ~IndirectGWAS(){};
    void run(std::vector<std::string> filenames, std::string output_stem);
};
