#ifndef __IO_H_
#define __IO_H_

#include <fstream>
#include <sstream>
#include <vector>

#include <Eigen/Dense>

using Eigen::Map;
using Eigen::Matrix;
using Eigen::RowMajor;

template <typename M>
M load_csv(std::string const &path, bool transpose = true) {
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<typename M::Scalar> values;
  int rows = 0;
  while (std::getline(indata, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      values.push_back(std::stod(cell));
    }
    ++rows;
  }
  indata.close();

  M result = Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime,
                              M::ColsAtCompileTime, RowMajor>>(
      values.data(), rows, (int)(values.size() / rows));

  if (transpose)
    return result.transpose();
  return result;
}

template <typename M>
void save_csv(std::string const &path, const M &data) {
  std::ofstream outdata(path);
  for (int i = 0; i < data.rows(); ++i) {
    outdata << data(i, 0);
    for (int j = 1; j < data.cols(); ++j) {
      outdata << "," << data(i, j);
    }
    outdata << std::endl;
  }
  outdata.close();
}

#endif // __IO_H_
