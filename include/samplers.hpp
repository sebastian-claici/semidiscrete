#ifndef __SAMPLERS_H_
#define __SAMPLERS_H_

#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>
#include <memory>
#include <random>

namespace semidiscrete {

/**
 * Abstract base class.
 *
 * Every subclass must implement two sampling methods: one which returns
 * a single vector sample from the distribution, and a second which returns
 * n samples as a d x n matrix.
 * */
class Distribution {
public:
  virtual Eigen::VectorXd operator()() const = 0;
  virtual Eigen::MatrixXd operator()(int n) const = 0;
};

/**
 * Multivariate normal distribution.
 * */
class MultivariateNormal : public Distribution {
private:
  Eigen::VectorXd mean;
  Eigen::MatrixXd transform;

public:
  /**
   * Mean 0 and identity covariance constructor.
   *
   * @param d int Dimension.
   * */
  MultivariateNormal(int d)
      : MultivariateNormal(Eigen::VectorXd::Zero(d),
                           Eigen::MatrixXd::Identity(d, d)) {}

  /**
   * Mean 0 constructor.
   *
   * @param covar MatrixXd Covariance matrix.
   * */
  MultivariateNormal(Eigen::MatrixXd const &covar)
      : MultivariateNormal(Eigen::VectorXd::Zero(covar.rows()), covar) {}

  /**
   * General constructor with given mean and covariance.
   *
   * @param mean VectorXd Mean of the distribution.
   * @param covar MatrixXd Covariance matrix.
   * */
  MultivariateNormal(Eigen::VectorXd const &mean, Eigen::MatrixXd const &covar)
      : mean(mean) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
    transform = eigenSolver.eigenvectors() *
                eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
  }

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

/**
 * Data distribution.
 * */
class DataDistribution : public Distribution {
private:
  Eigen::MatrixXd data;
  int n, d;
  std::vector<double> weights;

public:
  DataDistribution(Eigen::MatrixXd const &data)
      : data(data), n(data.cols()), d(data.rows()) {
    weights = std::vector<double>(n, 1.);
  }

  template <typename WeightIterator>
  DataDistribution(Eigen::MatrixXd const &data,
                   WeightIterator begin, WeightIterator end)
      : data(data), n(data.cols()), d(data.rows()), weights(begin, end) {}

  Eigen::VectorXd operator()() const {
    static std::mt19937 gen{std::random_device{}()};
    static std::discrete_distribution<int> dist(weights.begin(), weights.end());

    return data.col(dist(gen));
  }

  Eigen::MatrixXd operator()(int num) const {
    static std::mt19937 gen{std::random_device{}()};
    static std::discrete_distribution<int> dist(weights.begin(), weights.end());

    Eigen::MatrixXd output{d, num};
    for (int i = 0; i < num; ++i) {
      int id = dist(gen);
      output.col(i) = data.col(id);
    }

    return output;
  }
};

/**
 * Mixture distribution.
 *
 * Samples from the input distributions (D1, ..., Dn) with probability
 * proportional to the weight vector (w1, ..., wn).
 *
 * This class accepts polymorphic distributions, and so it stores pointers to
 * the distribution objects. In particular, this means that the constructor must
 * be called with the iterators to a data structure that also holds pointers to
 * these objects.
 *
 * For example, if we have two MultivariateNormal distributions mu and nu, we
 * can create a vector to hold them by std::vector<Distribution*> dists{&mu,
 * &nu}.
 * */
class Mixture : public Distribution {
private:
  std::vector<Distribution *> distributions;
  std::vector<double> weights;

public:
  template <typename DistIterator, typename WeightsIterator>
  Mixture(DistIterator d_begin, DistIterator d_end, WeightsIterator w_begin,
          WeightsIterator w_end)
      : distributions(d_begin, d_end), weights(w_begin, w_end) {}

  Eigen::VectorXd operator()() const {
    static std::mt19937 gen{std::random_device{}()};
    static std::discrete_distribution<int> dist( weights.begin(), weights.end() );

    int choice = dist(gen);
    return distributions[choice]->operator()();
  }

  Eigen::MatrixXd operator()(int num) const {
    static std::mt19937 gen{std::random_device{}()};
    static std::discrete_distribution<int> dist( weights.begin(), weights.end() );

    int d = distributions[0]->operator()().size();
    Eigen::MatrixXd output{d, num};
    for (int i = 0; i < num; ++i) {
      int choice = dist(gen);
      output.col(i) = distributions[choice]->operator()();
    }

    return output;
  }
};
}; // namespace semidiscrete

#endif // __SAMPLERS_H
