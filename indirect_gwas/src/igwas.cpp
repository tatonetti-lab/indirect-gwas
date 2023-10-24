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
    ensure_names_consistent(projection_coefficients.row_names, feature_partial_variance.names);

    // Compute the number of features and projections
    n_features = feature_partial_variance.names.size();
    n_projections = projection_coefficients.data.cols();

    // Initialize the chunk matrices
    degrees_of_freedom = {Eigen::VectorXd::Zero(0), variant_ids};
    beta = {Eigen::MatrixXd::Zero(0, n_projections), variant_ids, projection_coefficients.column_names};
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

std::unordered_map<std::string, unsigned int> IndirectGWAS::get_header_indexes(std::string header_line)
{
    std::istringstream iss(header_line);
    std::unordered_map<std::string, unsigned int> results;
    int i = 0;
    std::string header;
    while (std::getline(iss, header, ','))
    {
        if (header == column_names.variant_id_column ||
            header == column_names.beta_column ||
            header == column_names.se_column ||
            header == column_names.sample_size_column)
        {
            results[header] = i;
        }

        i++;
    }

    // Check that all four columns were found. If not, print some details and throw an error.
    if (results.size() != 4)
    {
        std::cerr << "Error: could not find all four columns in the header line" << std::endl;
        std::cerr << "Header line: " << header_line << std::endl;
        std::cerr << "Variant ID column: " << column_names.variant_id_column << std::endl;
        std::cerr << "Beta column: " << column_names.beta_column << std::endl;
        std::cerr << "Standard error column: " << column_names.se_column << std::endl;
        std::cerr << "Sample size column: " << column_names.sample_size_column << std::endl;
        throw std::runtime_error("Error: could not find all four columns in the header line");
    }

    return results;
}

void IndirectGWAS::read_file_chunk(
    std::string filename,
    unsigned int start_row,
    unsigned int end_row)
{
    // Read the file
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Error: could not open file " + filename);

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
            break;

        // TODO: Could greatly optimize this by not reading the entire line
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
    Eigen::VectorXd denominator = new_dof.array() * std_error_vec.array().square() + beta_vec.array().square();
    gpv_sum.data += (denominator.cwiseInverse() * feature_partial_variance.data[k]);
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
        }
    }
    se = se.cwiseSqrt();
}

void IndirectGWAS::compute_p_value(ResultsChunk &results) {
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
