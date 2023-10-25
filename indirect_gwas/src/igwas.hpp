#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "io.hpp"
#include "utils.hpp"

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
    void ensure_names_consistent(std::vector<std::string> names_1, std::vector<std::string> names_2);
    void read_file_chunk(std::string filename, unsigned int start_row, unsigned int end_row);
    void process_file_chunk(unsigned int k, std::string filename, unsigned int start_row, unsigned int end_row);
    void compute_standard_error(ResultsChunk &results);
    void compute_p_value(ResultsChunk &results);
    ResultsChunk compute_results_chunk();
    void reset_running_data(unsigned int chunksize);
    void save_results_chunk(ResultsChunk &results, std::string output_stem, bool write_header);

public:
    IndirectGWAS(
        ColumnSpec column_names,
        LabeledMatrix projection_coefficients,
        LabeledMatrix feature_partial_covariance,
        const unsigned int n_covariates,
        const unsigned int chunksize);
    ~IndirectGWAS(){};
    void run(std::vector<std::string> filenames, std::string output_stem);
};
