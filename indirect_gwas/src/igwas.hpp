#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "io.hpp"
#include "utils.hpp"

struct IndexedGwasChunk
{
    unsigned int feature_index;
    FeatureGwasResults results;
};

class IndirectGWAS
{
private:
    // Names of columns in the input files
    ColumnSpec column_names;

    // Parameters
    const unsigned int n_covariates;
    const unsigned int chunksize;
    bool single_file_output;

    // Input data
    LabeledMatrix projection_coefficients;
    LabeledVector feature_partial_variance;

    // Empty vectors to hold data from each chunk
    std::vector<std::string> variant_ids;

    // Computed data
    LabeledVector projection_partial_variance;
    LabeledVector degrees_of_freedom;
    LabeledMatrix beta;
    LabeledVector gpv_sum;
    unsigned int n_features;
    unsigned int n_projections;

    // Internal helpers
    std::queue<IndexedGwasChunk> results_queue;
    std::mutex results_mutex;
    std::condition_variable cv;
    bool finished_reading = false;

    // Internal helper functions
    void ensure_names_consistent(std::vector<std::string> names_1, std::vector<std::string> names_2);
    void reset_running_data(unsigned int chunksize);

    // Functions for threaded processing of feature GWAS files
    void load_chunk(std::vector<std::string> filenames, unsigned int chunk_start, unsigned int chunk_end);
    void raw_results_producer(std::vector<std::string> filenames, unsigned int chunk_start,
                              unsigned int chunk_end);
    void raw_results_consumer();

    // Update running sufficient statistics
    void process_file_chunk(unsigned int k, FeatureGwasResults &results);

    // Final results computation
    void compute_standard_error(ResultsChunk &results);
    void compute_p_value(ResultsChunk &results);
    ResultsChunk compute_results_chunk();

    // File output
    void save_results_chunk(ResultsChunk &results, std::string output_stem, bool write_header);
    void save_results_single_file(ResultsChunk &results, std::string output_stem, bool write_header);

public:
    IndirectGWAS(
        ColumnSpec column_names,
        LabeledMatrix projection_coefficients,
        LabeledMatrix feature_partial_covariance,
        const unsigned int n_covariates,
        const unsigned int chunksize,
        bool single_file_output);
    ~IndirectGWAS(){};
    void run(std::vector<std::string> filenames, std::string output_stem);
};
