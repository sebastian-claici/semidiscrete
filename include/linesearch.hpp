#ifndef __LINESEARCH_H_
#define __LINESEARCH_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>

using Eigen::Dynamic;
using Eigen::Matrix;

namespace semidiscrete {
class BacktrackingLinesearch {
private:
  double alpha;
  double c;
  double rho;
  int maxiters;

public:
  BacktrackingLinesearch(double alpha_, double c_, double rho_, int maxiters_)
      : alpha(alpha_), c(c_), rho(rho_), maxiters(maxiters_) {}

  template <typename Functor>
  double operator()(Functor &f, Matrix<double, Dynamic, 1> &x, double &fx,
                    Matrix<double, Dynamic, 1> &gx) {
    int iter = 0;
    Matrix<double, Dynamic, 1> gnew(x.size());
    auto fnew = f(x - alpha * gx, gnew);
    while (fnew < fx - c * alpha * gx.dot(gx) && iter < maxiters) {
      alpha = rho * alpha;
      iter += 1;
    }
    return alpha;
  }
};

class DampenedLinesearch {
private:
  double tau;
  int it;

public:
  DampenedLinesearch(double tau_) : tau(tau_), it(0) {}

  template <typename Functor>
  double operator()(Functor &f, Matrix<double, Dynamic, 1> &x, double &fx,
                    Matrix<double, Dynamic, 1> &gx) {
    it += 1;
    return tau * 1. / std::sqrt(it);
  }
};


}; // namespace semidiscrete

#endif // __LINESEARCH_H_
