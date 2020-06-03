#include "functors.hpp"
#include "linesearch.hpp"
#include "optimize.hpp"
#include "params.hpp"
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
  int num_points = 20;
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

  // Setup Anderson acceleration parameters and create solver
  AndersonParams params;
  params.max_iterations = 200;
  params.max_time = 20.0;
  DampenedLinesearch linesearch(10.0);
  AndersonSolver<DampenedLinesearch> solver(params, linesearch);

  VectorXd gw(weights.size());
  double fx = problem(weights, gw);
  std::cout << "gradient norm (at start) = " << gw.norm() << std::endl << std::endl;

  int iters = solver.minimize(problem, weights, fx);

  problem(weights, gw);
  std::cout << iters << " iterations" << std::endl;
  std::cout << "gradient norm = " << gw.norm() << std::endl;
  std::cout << "weights = \n" << weights.transpose() << std::endl;

  return 0;
}
