#include "igwas.hpp"

IndirectGWAS::IndirectGWAS(
    ColumnSpec column_names,
    std::string projection_coefficients_filename,
    std::string feature_partial_covariance_filename,
    const unsigned int n_covariates,
    const unsigned int chunksize) : column_names(column_names),
                                    n_covariates(n_covariates),
                                    chunksize(chunksize),
                                    projection_coefficients(read_input_matrix(projection_coefficients_filename))
{
    // Load the full feature partial covariance matrix
    LabeledMatrix feature_partial_covariance = read_input_matrix(feature_partial_covariance_filename);

    // Extract the diagonal of the feature partial covariance matrix
    feature_partial_variance = {feature_partial_covariance.data.diagonal(), feature_partial_covariance.row_names};

    // Compute the partial variance of each projection
    projection_partial_variance = {
        (projection_coefficients.data.transpose() * feature_partial_covariance.data * projection_coefficients.data).diagonal(),
        projection_coefficients.column_names};

    // Check that the features are the same in both files
    if (projection_coefficients.row_names != feature_partial_variance.names)
    {
        std::cerr << "Error: feature names do not match between files" << std::endl;

        std::cerr << "Projection coefficients names: ";
        for (auto name : projection_coefficients.row_names)
            std::cerr << name << " ";
        std::cerr << std::endl;

        std::cerr << "Feature partial variance names: ";
        for (auto name : feature_partial_variance.names)
            std::cerr << name << " ";
        std::cerr << std::endl;

        throw std::runtime_error("Feature names do not match");
    }

    // Compute the number of features and projections
    n_features = feature_partial_variance.names.size();
    n_projections = projection_coefficients.data.cols();

    // Initialize the chunk matrices
    degrees_of_freedom = {Eigen::VectorXd::Zero(0), variant_ids};
    beta = {Eigen::MatrixXd::Zero(0, n_projections), variant_ids, projection_coefficients.column_names};
    gpv_sum = {Eigen::VectorXd::Zero(0), variant_ids};
}

std::unordered_map<std::string, unsigned int> IndirectGWAS::get_header_indexes(std::string header_line)
{
    std::istringstream iss(header_line);
    std::vector<std::string> headers;
    std::string header;
    while (std::getline(iss, header, ','))
    {
        headers.push_back(header);
    }

    int variant_id_index = -1, beta_index = -1, se_index = -1, sample_size_index = -1;
    for (int i = 0; i < headers.size(); i++)
    {
        if (headers[i] == column_names.variant_id_column)
        {
            variant_id_index = i;
        }
        else if (headers[i] == column_names.beta_column)
        {
            beta_index = i;
        }
        else if (headers[i] == column_names.se_column)
        {
            se_index = i;
        }
        else if (headers[i] == column_names.sample_size_column)
        {
            sample_size_index = i;
        }
    }

    // Do not allow missing columns
    if (variant_id_index < 0)
    {
        throw std::runtime_error("Could not find column " + column_names.variant_id_column);
    }
    if (beta_index < 0)
    {
        throw std::runtime_error("Could not find column " + column_names.beta_column);
    }
    if (se_index < 0)
    {
        throw std::runtime_error("Could not find column " + column_names.se_column);
    }
    if (sample_size_index < 0)
    {
        throw std::runtime_error("Could not find column " + column_names.sample_size_column);
    }

    return {
        {column_names.variant_id_column, variant_id_index},
        {column_names.beta_column, beta_index},
        {column_names.se_column, se_index},
        {column_names.sample_size_column, sample_size_index}};
}

void IndirectGWAS::read_file_chunk(std::string filename, unsigned int start_row, unsigned int end_row)
{
    // Read the file
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Error: could not open file " + filename);

    // Read the header line
    std::string line;
    std::getline(file, line);
    auto header_indexes = get_header_indexes(line);

    // Read the data into vectors
    bool fill_variant_ids = variant_ids.size() == 0;
    unsigned int n_rows = end_row - start_row;
    beta_vec.resize(n_rows);
    std_error_vec.resize(n_rows);
    sample_size_vec.resize(n_rows);
    unsigned int row_index = 0;
    while (std::getline(file, line))
    {
        if (row_index < start_row)
        {
            row_index++;
            continue;
        }
        if (row_index > end_row)
        {
            break;
        }
        std::istringstream iss(line);
        std::vector<std::string> data;
        std::string datum;
        while (std::getline(iss, datum, ','))
        {
            data.push_back(datum);
        }

        // Populate variant IDs or check that they match
        if (fill_variant_ids)
        {
            variant_ids.push_back(data[header_indexes[column_names.variant_id_column]]);
        }
        else
        {
            if (variant_ids[row_index - start_row] != data[header_indexes[column_names.variant_id_column]])
            {
                throw std::runtime_error("Error: variant IDs do not match between files");
            }
        }

        // Populate the data vectors
        beta_vec(row_index - start_row) = std::stod(data[header_indexes[column_names.beta_column]]);
        std_error_vec(row_index - start_row) = std::stod(data[header_indexes[column_names.se_column]]);
        sample_size_vec(row_index - start_row) = std::stoi(data[header_indexes[column_names.sample_size_column]]);
        row_index++;
    }
}

void IndirectGWAS::process_file_chunk(unsigned int k, std::string filename, unsigned int start_row, unsigned int end_row)
{
    // Read the data
    read_file_chunk(filename, start_row, end_row);

    // Update the degrees of freedom as the minimum of the current degrees of freedom
    // and the current (sample size - number of exogenous variables - 1)
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
    Eigen::VectorXd denominator = new_dof.array() * std_error_vec.array().square() + beta_vec.array().square();
    gpv_sum.data += (denominator.cwiseInverse() * feature_partial_variance.data[k]);
}

ResultsChunk IndirectGWAS::compute_results_chunk()
{
    ResultsChunk results;

    // Set the variant IDs, avoiding a copy
    results.variant_ids = std::move(variant_ids);

    // Create a reference to gpv_sum that is called gpv_mean (to avoid ambiguity)
    Eigen::VectorXd &gpv_mean = gpv_sum.data;
    gpv_mean = gpv_mean.array() / n_features;

    // Compute the standard error of the beta coefficients
    results.std_error = -beta.data.array().square();
    for (int i = 0; i < results.std_error.rows(); i++)
    {
        for (int j = 0; j < results.std_error.cols(); j++)
        {
            results.std_error(i, j) += projection_partial_variance.data(j) / gpv_mean(i);
            results.std_error(i, j) /= degrees_of_freedom.data(i);
        }
    }
    results.std_error = results.std_error.cwiseSqrt();

    // Compute the t-statistic
    results.t_statistic = beta.data.array() / results.std_error.array();

    // Move beta to results, avoiding a copy
    results.beta = std::move(beta.data);

    // Compute the p-value by calling compute_log_p_value on each element of the t-statistic matrix
    results.neg_log10_p_value.resizeLike(results.t_statistic);

    for (int i = 0; i < results.t_statistic.rows(); i++)
    {
        for (int j = 0; j < results.t_statistic.cols(); j++)
        {
            if (degrees_of_freedom.data(i) <= 0 || std::isnan(degrees_of_freedom.data(i)))
            {
                throw std::runtime_error("Degrees of freedom is an error for variant " +
                                         results.variant_ids[i] + " with value " +
                                         std::to_string(degrees_of_freedom.data(i)));
            }
            results.neg_log10_p_value(i, j) = compute_log_p_value(results.t_statistic(i, j), degrees_of_freedom.data(i, 1));
        }
    }

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
            file << "variant_id,beta,std_error,t_statistic,neg_log10_p_value,sample_size"
                 << std::endl;
        }
        else
        {
            file.open(filename, std::ios_base::app);
        }

        // Write the results
        for (int j = 0; j < results.variant_ids.size(); j++)
        {
            // IDEA: Could set precision
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

void IndirectGWAS::run(std::vector<std::string> filenames, std::string output_stem)
{
    // Count the number of lines to ensure appropriate chunking
    unsigned int n_lines = count_lines(filenames[0]) - 1;

    // Iterate across all chunks
    unsigned int chunk_start_line = 0;
    while (chunk_start_line < n_lines)
    {
        unsigned int chunk_end_line = std::min(chunk_start_line + chunksize - 1, n_lines);
        unsigned int this_chunksize = chunk_end_line - chunk_start_line;

        // Clear the variant IDs
        variant_ids.clear();

        // Initialize the chunk matrices
        degrees_of_freedom.data.resize(this_chunksize);
        beta.data.resize(this_chunksize, n_projections);
        gpv_sum.data.resize(this_chunksize, n_projections);

        // Iterate across all files, updating running summary statistics
        for (int i = 0; i < filenames.size(); i++)
        {
            process_file_chunk(i, filenames[i], chunk_start_line, chunk_end_line);
        }

        // Compute the final results for this chunk
        ResultsChunk results = compute_results_chunk();

        // Save the results for this chunk
        save_results_chunk(results, output_stem, chunk_start_line == 0);

        // Update the chunk start line
        chunk_start_line = chunk_end_line + 1;
    }
}
