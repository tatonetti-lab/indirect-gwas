#include "io.hpp"

LabeledMatrix read_input_matrix(std::string filename)
{
    // Open the file
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Error: could not open file");

    // Read the data into a vector of vectors
    std::vector<std::vector<double>> data;
    std::vector<std::string> row_names;
    std::string line;
    std::getline(file, line); // Read the first line to get column names
    std::istringstream iss(line);
    std::string column_name;
    std::vector<std::string> column_names;

    // Throw away the first column name (if it exists)
    std::getline(iss, column_name, ',');

    while (std::getline(iss, column_name, ','))
    {
        column_names.push_back(column_name);
    }
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string row_name;
        std::getline(iss, row_name, ',');
        row_names.push_back(row_name);
        std::vector<double> row;
        double value;
        while (iss >> value)
        {
            row.push_back(value);
            if (iss.peek() == ',') iss.ignore();
        }
        data.push_back(row);
    }

    // Convert the vector of vectors to a matrix
    Eigen::MatrixXd matrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); i++)
    {
        for (int j = 0; j < data[i].size(); j++)
        {
            matrix(i, j) = data[i][j];
        }
    }

    // Check that sizes are consistent
    if (row_names.size() != matrix.rows())
    {
        throw std::runtime_error("Error: number of row names does not match number of rows");
    }
    if (column_names.size() != matrix.cols())
    {
        throw std::runtime_error("Error: number of column names does not match number of columns");
    }

    // Return the labeled matrix
    return LabeledMatrix{matrix, row_names, column_names};
}

LabeledVector read_input_vector(std::string filename)
{
    // Open the file
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Error: could not open file");

    // Read the data into a vector
    std::vector<double> data;
    std::vector<std::string> names;
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string name;
        std::getline(iss, name, ',');
        names.push_back(name);
        double value;
        iss >> value;
        data.push_back(value);
    }

    // Convert the vector to a vector
    Eigen::VectorXd vector(data.size());
    for (int i = 0; i < data.size(); i++)
    {
        vector(i) = data[i];
    }

    // Check that sizes are consistent
    if (names.size() != vector.size())
    {
        throw std::runtime_error("Error: number of names does not match number of elements");
    }

    // Return the labeled vector
    return LabeledVector{vector, names};
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
