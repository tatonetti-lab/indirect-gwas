#include "io.hpp"

LabeledMatrix read_input_matrix(std::string filename)
{
    csv::CSVReader reader(filename);

    LabeledMatrix result;
    result.column_names = reader.get_col_names();
    result.column_names.erase(result.column_names.begin());

    std::vector<std::vector<double>> temp_data;
    for (csv::CSVRow &row : reader)
    {
        int col_index = 0;
        std::vector<double> temp_row;
        for (csv::CSVField &field : row)
        {
            if (col_index == 0)
            {
                result.row_names.push_back(field.get<std::string>());
                col_index++;
                continue;
            }
            temp_row.push_back(field.get<double>());
        }
        temp_data.push_back(temp_row);
    }

    result.data.resize(temp_data.size(), temp_data[0].size());
    for (int i = 0; i < temp_data.size(); i++)
    {
        for (int j = 0; j < temp_data[0].size(); j++)
        {
            result.data(i, j) = temp_data[i][j];
        }
    }

    return result;
}

LabeledVector read_input_vector(std::string filename)
{
    LabeledVector result;
    std::vector<double> temp_data;

    csv::CSVReader reader(filename, csv::CSVFormat().no_header());

    for (csv::CSVRow &row : reader)
    {
        result.names.push_back(row[0].get<std::string>());
        temp_data.push_back(row[1].get<double>());
    }

    result.data.resize(temp_data.size());
    for (int i = 0; i < temp_data.size(); i++)
    {
        result.data(i) = temp_data[i];
    }

    return result;
}

unsigned int count_lines(std::string filename)
{
    // Count the number of lines in the first file
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Error: could not open file");
    }
    unsigned int n_lines = 0;
    std::string line;
    while (std::getline(file, line))
    {
        n_lines++;
    }
    return n_lines;
}
