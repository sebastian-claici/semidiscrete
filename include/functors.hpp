#ifndef __FUNCTORS_H_
#define __FUNCTORS_H_

#include <Eigen/Core>

#include <cassert>
#include <iostream>
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

  MatrixXd points;
  VectorXd weights;

  int num_samples;
  int minibatch_size;
public:

  /**
   * Construct a semidiscrete transport problem.
   *
   * @param sampler Sampler Data distribution to sample from.
   * @param points MatrixXd Point set of finite distribution.
   * */
  SemidiscreteTransport(const Sampler &sampler_,
                        const MatrixXd &points_,
                        const int num_samples_=1000,
                        const int minibatch_size_=100)
    : sampler(sampler_)
    , points(points_)
    , weights(VectorXd::Ones(points.cols()))
    , num_samples(num_samples_), minibatch_size(minibatch_size_) {
    assert(num_samples > 0);
    assert(minibatch_size > 0);
    assert(minibatch_size < num_samples);

    // normalize weights to sum 1
    weights /= weights.sum();
  }

  /**
   * Construct a semidiscrete transport problem.
   *
   * @param sampler Sampler Data distribution to sample from.
   * @param points MatrixXd Point set of finite distribution.
   * @param weights VectorXd Probability of each point in points.
   * */
  SemidiscreteTransport(const Sampler &sampler_,
                        const MatrixXd &points_, const VectorXd &weights_,
                        const int num_samples_=1000, const int minibatch_size_=100)
    : sampler(sampler_)
    , points(points_)
    , weights(weights_)
    , num_samples(num_samples_)
    , minibatch_size(minibatch_size_) {
    assert(points.cols() == weights.size());
    assert(num_samples > 0);
    assert(minibatch_size > 0);
    assert(minibatch_size < num_samples);

    // normalize weights to sum 1
    weights /= weights.sum();
  }

  /**
   * Compute the transport cost and gradient with given weights.
   *
   * @param w VectorXd Power cell weights, w_i is associated to x_i.
   * @param grad VectorXd Gradient given by -w_i + density(x_i)
   * */
  double operator()(const VectorXd &v, VectorXd &grad) {
    double fx = 0.0;
    int num_points = points.cols();
    for (int i = 0; i < num_points; ++i) {
      grad[i] = -weights(i);
      fx -= weights(i) * v(i);
    }

    for (int i = 0; i < num_samples / minibatch_size; ++i) {
      auto ps = sampler(minibatch_size);

      for (int bi = 0; bi < minibatch_size; ++bi) {
        Eigen::VectorXd::Index min_pos;
        auto dist = ((points.colwise() - ps.col(bi)).colwise().squaredNorm() -
                     v.transpose())
                        .minCoeff(&min_pos);
        fx += dist / num_samples;
        grad[static_cast<int>(min_pos)] += 1.0 / num_samples;
      }
    }

    return fx;
  }
};
}; // namespace semidiscrete

#endif // __FUNCTORS_H_
