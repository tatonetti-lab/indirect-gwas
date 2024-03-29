#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "igwas.hpp"
#include "io.hpp"
#include "utils.hpp"

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
    const unsigned int chunksize,
    bool single_file_output)
{
    IndirectGWAS igwas_obj{
        ColumnSpec{variant_id_column, beta_column, std_error_column, sample_size_column},
        read_input_matrix(projection_coefficients_filename),
        read_input_matrix(feature_partial_covariance_filename),
        n_covariates,
        chunksize,
        single_file_output};

    igwas_obj.run(feature_gwas_summary_filenames, output_stem);
}

PYBIND11_MODULE(_igwas, m)
{
    m.def(
        "run",
        &run_analysis,
        "Runs indirect GWAS using the C++ backend.\n\n"
        "Parameters\n----------\n"
        "feature_gwas_summary_filenames : list[str]\n"
        "    List of filenames of feature GWAS summary statistics files. These files\n"
        "    should be in the same order as the features in the projection coefficients\n"
        "    file and the feature partial covariance file. Files should have a header\n"
        "    row that contains, at minimum, the variant ID column, the effect size\n"
        "    column, the standard error column, and the sample size column.\n"
        "variant_id_column : str\n"
        "    Name of column in feature GWAS summary files containing variant IDs.\n"
        "beta_column : str\n"
        "    Name of column in feature GWAS summary files containing effect sizes.\n"
        "std_error_column : str\n"
        "    Name of column in feature GWAS summary files containing standard errors.\n"
        "sample_size_column : str\n"
        "    Name of column in feature GWAS summary files containing sample sizes.\n"
        "projection_coefficients_filename : str\n"
        "    Filename of projection coefficients file. This file should have a header\n"
        "    row containing projection names, and the first column should contain\n"
        "    feature names.\n"
        "feature_partial_covariance_filename : str\n"
        "    Filename of feature partial covariance file. This file should have a\n"
        "    header row and column, both containing feature names, which should be\n"
        "    the same as the feature names in the projection coefficients file.\n"
        "output_stem : str\n"
        "    Stem of output filenames.\n"
        "n_covariates : int\n"
        "    Number of covariates used in the feature GWAS.\n"
        "chunksize : int\n"
        "    Number of rows to read from feature GWAS summary files at a time.\n"
        "single_file_output : bool\n"
        "    Whether to write all results to a single file.\n\n"
        "Returns\n-------\n"
        "None\n\n",
        py::arg("feature_gwas_summary_filenames"),
        py::arg("variant_id_column"),
        py::arg("beta_column"),
        py::arg("std_error_column"),
        py::arg("sample_size_column"),
        py::arg("projection_coefficients_filename"),
        py::arg("feature_partial_covariance_filename"),
        py::arg("output_stem"),
        py::arg("n_covariates"),
        py::arg("chunksize"),
        py::arg("single_file_output"));
    m.def("compute_pvalue", &compute_log_p_value, "Compute negative log10 p-value",
          py::arg("t_statistic"), py::arg("degrees_of_freedom"));
    m.def("head", &head, "Print the first n lines of a file", py::arg("filename"), py::arg("n_lines"));
}
