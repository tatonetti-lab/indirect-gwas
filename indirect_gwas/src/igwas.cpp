#include "igwas.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <csv.hpp>
#include <Eigen/Dense>

#include "io.hpp"
#include "utils.hpp"

IndirectGWAS::IndirectGWAS(
    ColumnSpec column_names,
    LabeledMatrix projection_coefficients,
    LabeledMatrix feature_partial_covariance,
    const unsigned int n_covariates,
    const unsigned int chunksize,
    bool single_file_output) : column_names(column_names),
                               n_covariates(n_covariates),
                               chunksize(chunksize),
                               projection_coefficients(projection_coefficients),
                               single_file_output(single_file_output)
{
    auto &projection_names = projection_coefficients.column_names;
    auto &feature_names = feature_partial_covariance.row_names;

    // Check that the features are the same in both files
    ensure_names_consistent(projection_coefficients.row_names, feature_names);

    // Extract the diagonal of the feature partial covariance matrix
    auto &C = feature_partial_covariance.data;
    feature_partial_variance = {C.diagonal(), feature_names};

    // Compute the partial variance of each projection
    auto &P = projection_coefficients.data;
    projection_partial_variance = {(P.transpose() * C * P).diagonal(), projection_names};

    // Compute the number of features and projections
    n_features = feature_names.size();
    n_projections = projection_names.size();

    // Initialize the chunk matrices
    degrees_of_freedom = {Eigen::VectorXd::Zero(0), variant_ids};
    beta = {Eigen::MatrixXd::Zero(0, n_projections), variant_ids, projection_names};
    gpv_sum = {Eigen::VectorXd::Zero(0), variant_ids};
}

void IndirectGWAS::ensure_names_consistent(std::vector<std::string> names_1,
                                           std::vector<std::string> names_2)
{
    if (names_1 != names_2)
    {
        std::cerr << "Error: feature names do not match between files" << std::endl;

        std::cerr << "First group: ";
        for (auto name : names_1)
            std::cerr << name << " ";
        std::cerr << std::endl;

        std::cerr << "Second group: ";
        for (auto name : names_2)
            std::cerr << name << " ";
        std::cerr << std::endl;

        throw std::runtime_error("Feature names do not match");
    }
}

void IndirectGWAS::read_file_chunk(
    std::string filename,
    unsigned int start_row,
    unsigned int end_row)
{
    csv::CSVReader reader(filename);

    bool fill_variant_ids = variant_ids.size() == 0;

    unsigned int row_index = 0;
    for (csv::CSVRow &row : reader)
    {
        if (row_index < start_row)
        {
            row_index++;
            continue;
        }
        if (row_index > end_row)
            break;

        // Populate variant IDs or check that they match
        if (fill_variant_ids)
        {
            variant_ids.push_back(row[column_names.variant_id].get<std::string>());
        }
        else if (variant_ids[row_index - start_row] != row[column_names.variant_id].get<std::string>())
        {
            throw std::runtime_error("Error: variant IDs do not match between files");
        }

        // Populate the data vectors
        beta_vec(row_index - start_row) = row[column_names.beta].get<double>();
        std_error_vec(row_index - start_row) = row[column_names.se].get<double>();
        sample_size_vec(row_index - start_row) = row[column_names.sample_size].get<int>();
        row_index++;
    }
}

void IndirectGWAS::process_file_chunk(
    unsigned int k, std::string filename,
    unsigned int start_row,
    unsigned int end_row)
{
    // Read the data
    read_file_chunk(filename, start_row, end_row);

    // Update the degrees of freedom as the minimum of the current degrees of freedom
    // and the current (sample size - number of exogenous variables - 1).
    // Note that the number of exogenous is n_covariates + 1.
    Eigen::VectorXd new_dof = sample_size_vec.array() - n_covariates - 2;
    if (k == 0)
    {
        degrees_of_freedom.data = new_dof;
    }
    else
    {
        degrees_of_freedom.data = degrees_of_freedom.data.cwiseMin(new_dof);
    }

    // Update the beta matrix
    for (int i = 0; i < variant_ids.size(); i++)
    {
        for (int j = 0; j < projection_coefficients.row_names.size(); j++)
        {
            beta.data(i, j) += beta_vec(i) * projection_coefficients.data(k, j);
        }
    }

    // Update the sum of genotype partial variances
    Eigen::VectorXd denom = new_dof.array() * std_error_vec.array().square() + beta_vec.array().square();
    gpv_sum.data += (denom.cwiseInverse() * feature_partial_variance.data[k]);
}

// Split the standard error computation into a separate function for readability
void IndirectGWAS::compute_standard_error(ResultsChunk &results)
{
    // Create a reference to gpv_sum that is called gpv_mean (to avoid ambiguity)
    Eigen::VectorXd &gpv = gpv_sum.data;
    gpv /= n_features;

    // Reference to the standard error matrix
    Eigen::MatrixXd &se = results.std_error;

    se = -results.beta.array().square();
    for (int i = 0; i < se.rows(); i++)
    {
        for (int j = 0; j < se.cols(); j++)
        {
            se(i, j) += projection_partial_variance.data(j) / gpv(i);
            se(i, j) /= degrees_of_freedom.data(i);

            if (std::isnan(se(i, j)) || se(i, j) < 0)
            {
                std::cerr << "Standard error for variant "
                          << results.variant_ids[i] << " is " << se(i, j)
                          << ". gpv = " << gpv(i)
                          << ", projection_partial_variance = "
                          << projection_partial_variance.data(j)
                          << ", degrees_of_freedom = " << degrees_of_freedom.data(i)
                          << ", beta = " << results.beta(i, j) << std::endl;
            }
        }
    }
    se = se.cwiseSqrt();
}

void IndirectGWAS::compute_p_value(ResultsChunk &results)
{
    Eigen::MatrixXd &t = results.t_statistic;
    Eigen::VectorXd &dof = degrees_of_freedom.data;
    Eigen::MatrixXd &p = results.neg_log10_p_value;

    p.resizeLike(t);
    for (int i = 0; i < t.rows(); i++)
    {
        for (int j = 0; j < t.cols(); j++)
        {
            if (dof(i) <= 0 || std::isnan(dof(i)))
            {
                throw std::runtime_error("Degrees of freedom is an error for variant " +
                                         results.variant_ids[i] + " with value " +
                                         std::to_string(dof(i)));
            }
            if (std::isnan(t(i, j)))
            {
                std::cerr << "Erroneous T-statistic for variant "
                          << results.variant_ids[i] << " with value " << t(i, j)
                          << ". Corresponding beta is " << results.beta(i, j)
                          << ", standard error is " << results.std_error(i, j)
                          << std::endl;

                p(i, j) = std::nan("");
                continue;
            }
            p(i, j) = compute_log_p_value(t(i, j), dof(i));
        }
    }
}

ResultsChunk IndirectGWAS::compute_results_chunk()
{
    ResultsChunk results;

    results.variant_ids = std::move(variant_ids);
    results.beta = std::move(beta.data);

    compute_standard_error(results);

    // Compute the t-statistic
    results.t_statistic = results.beta.array() / results.std_error.array();

    // Compute the p-value by calling compute_log_p_value on each t-statistic
    compute_p_value(results);

    // Add the sample size to the results
    results.sample_size = degrees_of_freedom.data.array() + n_covariates + 2;

    return results;
};

void IndirectGWAS::save_results_chunk(ResultsChunk &results, std::string output_stem, bool write_header)
{
    for (int i = 0; i < results.beta.cols(); i++)
    {
        std::string projection_name = projection_coefficients.column_names[i];

        // Open the output file
        std::string filename = output_stem + "_" + projection_name + ".csv";
        std::ofstream file;

        if (write_header)
        {
            file.open(filename);

            // Write the header
            // IDEA: Could use the same column names as the input files
            file << "variant_id,beta,std_error,t_statistic,neg_log10_p_value,sample_size"
                 << std::endl;
        }
        else
        {
            file.open(filename, std::ios_base::app);
        }

        // IDEA: Could set precision as a parameter
        file << std::setprecision(6);

        // Write the results
        for (int j = 0; j < results.variant_ids.size(); j++)
        {
            // IDEA: Could use other separators
            file << results.variant_ids[j] << ","
                 << results.beta(j, i) << ","
                 << results.std_error(j, i) << ","
                 << results.t_statistic(j, i) << ","
                 << results.neg_log10_p_value(j, i) << ","
                 << results.sample_size(j) << std::endl;
        }
    }
}

// Exactly the same as save_results_chunk, except that it writes to a single file,
// with an additional column for the projection name.
// void IndirectGWAS::save_results_single_file(ResultsChunk &results, std::string output_stem, bool write_header)
// {
//     // Open the output file
//     std::string filename = output_stem + ".csv";
//     std::ofstream file;

//     if (write_header)
//     {
//         file.open(filename);

//         // Write the header
//         // IDEA: Could use the same column names as the input files
//         file << "projection_id,variant_id,beta,std_error,t_statistic,"
//              << "neg_log10_p_value,sample_size" << std::endl;
//     }
//     else
//     {
//         file.open(filename, std::ios_base::app);
//     }

//     // IDEA: Could set precision as a parameter
//     file << std::setprecision(6);

//     // Write the results
//     for (int vid = 0; vid < results.variant_ids.size(); vid++)
//     {
//         for (int pid = 0; pid < results.beta.cols(); pid++)
//         {
//             // IDEA: Could use other separators
//             file << projection_coefficients.column_names[pid] << ","
//                  << results.variant_ids[vid] << ","
//                  << results.beta(vid, pid) << ","
//                  << results.std_error(vid, pid) << ","
//                  << results.t_statistic(vid, pid) << ","
//                  << results.neg_log10_p_value(vid, pid) << ","
//                  << results.sample_size(vid) << std::endl;
//         }
//     }
// }

void IndirectGWAS::save_results_single_file(ResultsChunk &results, std::string output_stem, bool write_header)
{
    std::ios_base::sync_with_stdio(false);

    std::ostringstream oss;
    oss.precision(6);

    int vidSize = results.variant_ids.size();
    int pidSize = results.beta.cols();

    for (int vid = 0; vid < vidSize; vid++)
    {
        for (int pid = 0; pid < pidSize; pid++)
        {
            oss << projection_coefficients.column_names[pid] << ","
                << results.variant_ids[vid] << ","
                << results.beta(vid, pid) << ","
                << results.std_error(vid, pid) << ","
                << results.t_statistic(vid, pid) << ","
                << results.neg_log10_p_value(vid, pid) << ","
                << results.sample_size(vid) << "\n";
        }
    }

    // Open the output file
    std::string filename = output_stem + ".csv";
    std::ofstream file;

    if (write_header)
    {
        file.open(filename);
        file << "projection_id,variant_id,beta,std_error,t_statistic,"
             << "neg_log10_p_value,sample_size" << std::endl;
    }
    else
    {
        file.open(filename, std::ios_base::app);
    }
    file << oss.str();
}

// Resets running data containers to the current chunk size and zeros where necessary
void IndirectGWAS::reset_running_data(unsigned int chunksize)
{
    // Clear the variant IDs
    variant_ids.clear();

    // Resize and zero the vectors and matrices
    beta_vec.resize(chunksize);
    std_error_vec.resize(chunksize);
    sample_size_vec.resize(chunksize);

    degrees_of_freedom.data = Eigen::VectorXd::Zero(chunksize);
    beta.data = Eigen::MatrixXd::Zero(chunksize, n_projections);
    gpv_sum.data = Eigen::VectorXd::Zero(chunksize);
}

// Runs the indirect GWAS. Assumes that all input files are identically formatted,
// with the same column names and identical variants in the same order.
void IndirectGWAS::run(std::vector<std::string> filenames, std::string output_stem)
{
    // Count the number of lines to ensure appropriate chunking
    unsigned int n_lines = count_lines(filenames[0]) - 1;

    // Iterate across all chunks
    unsigned int chunk_start_line = 0;
    while (chunk_start_line < n_lines)
    {
        unsigned int chunk_end_line = std::min(chunk_start_line + chunksize - 1, n_lines - 1);

        // Reset the running data to the current chunk size and zero where necessary
        reset_running_data(chunk_end_line - chunk_start_line + 1);

        // Iterate across all files, updating running summary statistics
        for (int i = 0; i < filenames.size(); i++)
        {
            process_file_chunk(i, filenames[i], chunk_start_line, chunk_end_line);
        }

        // Compute the final results for this chunk
        ResultsChunk results = compute_results_chunk();

        // Save the results for this chunk
        if (single_file_output)
        {
            save_results_single_file(results, output_stem, chunk_start_line == 0);
        }
        else
        {
            save_results_chunk(results, output_stem, chunk_start_line == 0);
        }

        // Update the chunk start line
        chunk_start_line = chunk_end_line + 1;
    }
}
