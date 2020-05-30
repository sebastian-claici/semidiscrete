#ifndef __PARAMS_H_
#define __PARAMS_H_

namespace semidiscrete {
struct GradientParams {
  double epsilon;
  int max_iterations;
  double max_time;

  GradientParams(const double &epsilon_, const int &max_iterations_, const double& max_time_)
    : epsilon(epsilon_),
      max_iterations(max_iterations_),
      max_time(max_time_) {}

  GradientParams()
    : epsilon(static_cast<double>(1e-6)),
      max_iterations(1000),
      max_time(5.0) {}
};

struct AndersonParams {
  int m;
  double lambda;
  double epsilon;
  int max_iterations;
  double max_time;

  AndersonParams(const int &m_, const double &lambda_, const double &epsilon_,
                 const int &max_iterations_, const double &max_time_)
    : m(m_),
      lambda(lambda_),
      epsilon(epsilon_),
      max_iterations(max_iterations_),
      max_time(max_time_) {}

  AndersonParams()
    : m(8),
      lambda(static_cast<double>(0.01)),
      epsilon(static_cast<double>(1e-6)),
      max_iterations(100),
      max_time(5.0) {}
};

struct AdamParams {
  double alpha;
  double beta1, beta2;
  double epsilon;
  int max_iterations;
  double max_time;

  AdamParams()
    : alpha(0.0001)
    , beta1(0.9)
    , beta2(0.999)
    , epsilon(1e-6)
    , max_iterations(1000)
    , max_time(5.0) {}
};
}; // namespace semidiscrete

#endif //
