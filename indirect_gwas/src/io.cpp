#include "io.hpp"

#include <fstream>
#include <string>
#include <vector>

#include <csv.hpp>

#include "utils.hpp"

LabeledMatrix read_input_matrix(std::string filename)
{
    csv::CSVReader reader(filename);

    LabeledMatrix result;
    result.column_names = reader.get_col_names();
    result.column_names.erase(result.column_names.begin());

    std::vector<std::vector<double>> temp_data;
    for (csv::CSVRow &row : reader)
    {
        bool first_col = true;
        std::vector<double> temp_row;
        for (csv::CSVField &field : row)
        {
            if (first_col)
            {
                result.row_names.push_back(field.get<std::string>());
                first_col = false;
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

unsigned int count_lines(std::string filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Error: could not open file");
    unsigned int n_lines = 0;
    std::string line;
    while (std::getline(file, line))
        n_lines++;
    return n_lines;
}

void head(std::string filename, unsigned int n_lines)
{
    csv::CSVReader reader(filename);

    // Print the header
    for (auto &field : reader.get_col_names())
    {
        std::cout << field << " ";
    }
    std::cout << std::endl;

    unsigned int row_index = 0;
    for (csv::CSVRow &row : reader)
    {
        if (row_index > n_lines)
            break;

        for (csv::CSVField &field : row)
        {
            std::cout << field.get<std::string>() << " ";
        }
        std::cout << std::endl;

        row_index++;
    }
}
