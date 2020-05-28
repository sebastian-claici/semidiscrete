#ifndef __SAMPLERS_H_
#define __SAMPLERS_H_

#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>
#include <random>

namespace semidiscrete {
struct MultivariateNormal {
  MultivariateNormal(Eigen::MatrixXd const &covar)
      : MultivariateNormal(Eigen::VectorXd::Zero(covar.rows()), covar) {}

  MultivariateNormal(Eigen::VectorXd const &mean, Eigen::MatrixXd const &covar)
      : mean(mean) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
    transform = eigenSolver.eigenvectors() *
                eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
  }

  Eigen::VectorXd mean;
  Eigen::MatrixXd transform;

  Eigen::VectorXd operator()() const {
    static std::mt19937 gen{std::random_device{}()};
    static std::normal_distribution<double> dist;

    return mean + transform * Eigen::VectorXd{mean.size()}.unaryExpr(
                                  [&](auto x) { return dist(gen); });
  }

  Eigen::MatrixXd operator()(int n) const {
    static std::mt19937 gen{std::random_device{}()};
    static std::normal_distribution<double> dist;

    Eigen::MatrixXd m = transform * Eigen::MatrixXd{mean.size(), n}.unaryExpr(
                                        [&](auto x) { return dist(gen); });
    m.colwise() += mean;

    return m;
  }
};

struct MultivariateUniform {
  MultivariateUniform(int const &dim) : dim(dim) {}

  int dim;

  Eigen::VectorXd operator()() const {
    static std::mt19937 gen{std::random_device{}()};
    static std::uniform_real_distribution<double> dist;

    return Eigen::VectorXd{dim}.unaryExpr([&](auto x) { return dist(gen); });
  }

  Eigen::MatrixXd operator()(int n) const {
    static std::mt19937 gen{std::random_device{}()};
    static std::normal_distribution<double> dist;

    return Eigen::MatrixXd{dim, n}.unaryExpr([&](auto x) { return dist(gen); });
  }
};

struct DataUniform {
  DataUniform(Eigen::MatrixXd const &data)
      : data(data), n(data.cols()), d(data.rows()) {}

  Eigen::MatrixXd data;
  int n, d;

  Eigen::VectorXd operator()() const {
    static std::mt19937 gen{std::random_device{}()};
    static std::uniform_int_distribution<int> dist(0, n - 1);

    return data.col(dist(gen));
  }

  Eigen::MatrixXd operator()(int num) const {
    static std::mt19937 gen{std::random_device{}()};
    static std::uniform_int_distribution<int> dist(0, n - 1);

    Eigen::MatrixXd output{d, num};
    for (int i = 0; i < num; ++i) {
      int id = dist(gen);
      output.col(i) = data.col(id);
    }

    return output;
  }
};

template <typename Distribution> struct Mixture {
  Mixture(std::vector<Distribution> const &distributions,
          std::vector<double> const &weights)
      : distributions(distributions), weights(weights) {}

  std::vector<Distribution> distributions;
  std::vector<double> weights;

  Eigen::VectorXd operator()() const {
    static std::mt19937 gen{std::random_device{}()};
    static std::discrete_distribution<int> dist{weights.begin(), weights.end()};

    int choice = dist(gen);
    return distributions[choice]();
  }

  Eigen::MatrixXd operator()(int num) const {
    static std::mt19937 gen{std::random_device{}()};
    static std::discrete_distribution<int> dist{weights.begin(), weights.end()};

    int d = distributions[0]().rows();
    Eigen::MatrixXd output{d, num};
    for (int i = 0; i < num; ++i) {
      int choice = dist(gen);
      output.col(i) = distributions[choice]();
    }

    return output;
  }
};
}; // namespace semidiscrete

#endif // __SAMPLERS_H_
