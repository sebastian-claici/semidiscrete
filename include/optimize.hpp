#ifndef __OPTIMIZE_H_
#define __OPTIMIZE_H_

#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>
#include <chrono>

#include "linesearch.hpp"
#include "params.hpp"

using Eigen::Dynamic;
using Eigen::Matrix;

namespace semidiscrete {

template <class LineSearch = BacktrackingLinesearch>
class GradientSolver {
private:
  GradientParams params;
  LineSearch linesearch;

public:
  GradientSolver(const GradientParams &params_, LineSearch &linesearch_)
      : params(params_), linesearch(linesearch_) {}

  /**
   * Minimize f starting from initial guess x.
   *
   * This procedure implements gradient descent on f with initial guess x.
   * This function is not guaranteed to converge, so a maximum number of
   * iterations should be specified in the parameters.
   *
   * @tparam Functor a class with callable operator() returning function and
   * gradient.
   * @param f Functor class that exposes an operator() function.
   * @param x Initial guess as typed Eigen::Vector. Used to store the result.
   * @param fx Function value at optimum.
   *
   * @return Number of iterations until convergence.
   * */
  template <typename Functor>
  int minimize(Functor &f, Matrix<double, Dynamic, 1> &x, double &fx) {
    Matrix<double, Dynamic, 1> gx(x.size());

    // keep track of time spent
    auto tstart = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < params.max_iterations; ++it) {
      fx = f(x, gx);
      if (gx.norm() < params.epsilon) {
        std::cout << "Gradient norm at convergence: " << gx.norm() << std::endl;
        return it + 1;
      }

      x = x - linesearch(f, x, fx, gx) * gx;

      // break if time exceeds maximum allotted
      auto titer = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(titer - tstart).count();
      if (duration / 1000. > params.max_time) {
        return it + 1;
      }
      // break if gradient norm within tolerance
      if (gx.norm() < params.epsilon) {
        std::cout << "Gradient norm at convergence: " << gx.norm() << std::endl;
        return it + 1;
      }
    }

    return params.max_iterations;
  }
};

template <class LineSearch = BacktrackingLinesearch>
class AndersonSolver {
private:
  AndersonParams params;
  LineSearch linesearch;

public:
  AndersonSolver(const AndersonParams &params_, LineSearch &linesearch_)
      : params(params_), linesearch(linesearch_) {}

  /**
   * Minimize f starting from initial guess x.
   *
   * @tparam Functor a class with callable operator() returning function and
   * gradient.
   * @param f Functor class that exposes an operator() function.
   * @param x Initial guess as typed Eigen::Vector. Used to store the result.
   * @param fx Function value at optimum.
   *
   * @return Number of iterations until convergence.
   * */
  template <typename Functor>
  int minimize(Functor &f, Matrix<double, Dynamic, 1> &x, double &fx) {
    int d = x.size();
    int m = params.m;

    // keep track of time spent
    auto tstart = std::chrono::high_resolution_clock::now();

    Matrix<double, Dynamic, 1> gx(x.size());
    Matrix<double, Dynamic, Dynamic> I =
        Matrix<double, Dynamic, Dynamic>::Identity(m + 1, m + 1);
    for (int it = 0; it < params.max_iterations; ++it) {
      Matrix<double, Dynamic, Dynamic> U(d, m + 1);
      Matrix<double, Dynamic, Dynamic> X(d, m + 1);
      for (int i = 0; i < m + 1; ++i) {
        fx = f(x, gx);
        X.col(i) = x;
        U.col(i) = -linesearch(f, x, fx, gx) * gx;
        x = x - linesearch(f, x, fx, gx) * gx;

        auto titer = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(titer - tstart).count();
        if (duration / 1000. > params.max_time) {
          fx = f(x, gx);
          return it + 1;
        }
      }
      Matrix<double, Dynamic, Dynamic> A =
          U.transpose() * U + params.lambda * I;

      Matrix<double, Dynamic, 1> b = Matrix<double, Dynamic, 1>::Ones(m + 1);
      Matrix<double, Dynamic, 1> z = A.colPivHouseholderQr().solve(b);
      z /= z.sum();

      x *= 0;
      for (int i = 0; i < m + 1; ++i) {
        x += z(i) * X.col(i);
      }

      fx = f(x, gx);

      // break if time exceeds maximum allotted
      auto titer = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(titer - tstart).count();
      if (duration / 1000. > params.max_time) {
        return it + 1;
      }
      // break if gradient norm within tolerance
      if (gx.norm() < params.epsilon) {
        std::cout << "Gradient norm at convergence: " << gx.norm() << std::endl;
        return it + 1;
      }
    }

    return params.max_iterations;
  }
};

class AdamSolver {
private:
  AdamParams params;

public:
  AdamSolver(const AdamParams &params_)
    : params(params_) {}

  /**
   * Minimize f starting from initial guess x.
   *
   * @tparam Functor a class with callable operator() returning function and
   * gradient.
   * @param f Functor class that exposes an operator() function.
   * @param x Initial guess as typed Eigen::Vector. Used to store the result.
   * @param fx Function value at optimum.
   *
   * @return Number of iterations until convergence.
   * */
  template <typename Functor>
  int minimize(Functor &f, Matrix<double, Dynamic, 1> &x, double &fx) {
    Matrix<double, Dynamic, 1> mt = Matrix<double, Dynamic, 1>::Zero(x.size());
    Matrix<double, Dynamic, 1> vt = Matrix<double, Dynamic, 1>::Zero(x.size());

    double b1t = params.beta1;
    double b2t = params.beta2;
    Matrix<double, Dynamic, 1> gx(x.size());

    auto tstart = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < params.max_iterations; ++it) {
      fx = f(x, gx);
      // break if gradient norm within tolerance
      if (gx.norm() < params.epsilon) {
        std::cout << "Gradient norm at convergence: " << gx.norm() << std::endl;
        return it + 1;
      }

      mt = params.beta1 * mt + (1.0 - params.beta1) * gx;
      vt = params.beta2 * vt + (1.0 - params.beta2) * gx.cwiseProduct(gx);
      auto mh = mt / (1 - b1t);
      auto vh = vt / (1 - b2t);
      x = x - params.alpha * mh.cwiseQuotient(vh.cwiseSqrt());

      b1t = b1t * params.beta1;
      b2t = b2t * params.beta2;

      auto titer = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(titer - tstart).count();
      if (duration / 1000. > params.max_time) {
        return it + 1;
      }
    }

    return params.max_iterations;
  }
};
}; // namespace semidiscrete

#endif // __OPTIMIZE_H_
