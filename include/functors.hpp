#ifndef __FUNCTORS_H_
#define __FUNCTORS_H_

#include <Eigen/Core>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXf;

namespace semidiscrete {

/**
 * Implements gradients for the semidiscrete transport cost.
 *
 * This class implements the density and barycenter functions that are
 * used to solve semidiscrete optimal transportation problems. The approach
 * taken here relies on sampling from the power cells induced by the
 * discrete distribution.
 *
 * This approach should not be used if the continuous distribution is highly
 * concentrated as the power cells are then sensitive to small perturbations.
 *
 * */
template <typename Sampler>
class SemidiscreteTransport {
private:
  Sampler sampler;
  int num_samples;

public:
  MatrixXd points;

  SemidiscreteTransport(const Sampler &sampler_, const MatrixXd &points_,
                        const int num_samples_)
      : sampler(sampler_), num_samples(num_samples_), points(points_) {}

  /**
   * Compute the transport cost and gradient with given weights.
   *
   * @param w VectorXd Power cell weights, w_i is associated to x_i.
   * @param grad VectorXd Gradient given by -1/n + density(x_i)
   * */
  double operator()(const VectorXd &w, VectorXd &grad) {
    int num_points = points.cols();
    for (int i = 0; i < num_points; ++i) {
      grad[i] = -1.0 / num_points;
    }

    double fx = 0.0;
    for (int i = 0; i < num_samples; ++i) {
      auto p = sampler();

      Eigen::VectorXd::Index min_pos;
      auto dist =
          ((points.colwise() - p).colwise().squaredNorm() - w.transpose())
              .minCoeff(&min_pos);
      fx += dist / num_samples;
      grad[static_cast<int>(min_pos)] += 1.0 / num_samples;
    }

    return fx;
  }

  void densities(const VectorXd &w, std::vector<double> &density) {
    for (int i = 0; i < num_samples; ++i) {
      auto p = sampler();

      Eigen::VectorXf::Index min_pos;
      ((points.colwise() - p).colwise().squaredNorm() - w.transpose())
          .minCoeff(&min_pos);

      int pos = static_cast<int>(min_pos);
      density[pos] += 1.0 / num_samples;
    }
  }

  void barycenters(const VectorXd &w, MatrixXd &new_points,
                   std::vector<double> &density) {
    for (int i = 0; i < num_samples; ++i) {
      auto p = sampler();

      Eigen::VectorXf::Index min_pos;
      ((points.colwise() - p).colwise().squaredNorm() - w.transpose())
          .minCoeff(&min_pos);

      int pos = static_cast<int>(min_pos);
      new_points.col(pos) += p / num_samples;
      density[pos] += 1.0 / num_samples;
    }
  }
};
}; // namespace semidiscrete

#endif // __FUNCTORS_H_
