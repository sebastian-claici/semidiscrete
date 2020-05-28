#include "functors.hpp"
#include "linesearch.hpp"
#include "optimize.hpp"
#include "samplers.hpp"

#include <algorithm>
#include <chrono>
#include <vector>

#define DEBUG 1

using namespace semidiscrete;

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {
  // Compute the transport cost between 100 samples from a Gaussian and the distribution itself
  int num_points = 100;
  int num_samples = 10000;
  int num_dim = 2;

  // Want to approximate a Gaussian distribution
  VectorXd mu1 = VectorXd::Zero(num_dim);
  auto mu = MultivariateNormal(mu1, 1 * MatrixXd::Identity(num_dim, num_dim));

  // Initial point positions
  MatrixXd points = mu(num_points);
  VectorXd weights = VectorXd::Zero(num_points);
  // Define transport problem
  SemidiscreteTransport<MultivariateNormal> problem(mu, points, num_samples);

  // Setup Adam parameters and create solver
  AdamParams params;
  params.max_iterations = 200;
  AdamSolver solver(params);

  VectorXd gw(weights.size());
  double fx = problem(weights, gw);
  std::cout << "cost (at start) = " << fx << std::endl << std::endl;

  int iters = solver.minimize(problem, weights, fx);

  std::cout << iters << " iterations" << std::endl;
  std::cout << "weights = \n" << weights.transpose() << std::endl;
  std::cout << "cost = " << fx << std::endl;

  return 0;
}
