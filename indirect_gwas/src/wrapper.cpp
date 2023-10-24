#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "igwas.hpp"

namespace py = pybind11;

void run_analysis(
    std::vector<std::string> feature_gwas_summary_filenames,
    std::string variant_id_column,
    std::string beta_column,
    std::string std_error_column,
    std::string sample_size_column,
    std::string projection_coefficients_filename,
    std::string feature_partial_covariance_filename,
    std::string output_stem,
    const unsigned int n_covariates,
    const unsigned int chunksize
    )
{
    IndirectGWAS igwas_obj{
        ColumnSpec{variant_id_column, beta_column, std_error_column, sample_size_column},
        projection_coefficients_filename,
        feature_partial_covariance_filename,
        n_covariates,
        chunksize
    };

    igwas_obj.run(feature_gwas_summary_filenames, output_stem);
}

PYBIND11_MODULE(_igwas, m)
{
    // m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("run", &run_analysis, "Run igwas");  // TODO: Add arguments and types
    m.def("compute_pvalue", &compute_log_p_value, "Compute p-value");
}
