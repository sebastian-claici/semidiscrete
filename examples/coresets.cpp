#include "io.hpp"
#include "functors.hpp"
#include "linesearch.hpp"
#include "optimize.hpp"
#include "samplers.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <vector>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#define DEBUG 1

using namespace semidiscrete;
namespace po = boost::program_options;

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(int argc, const char *argv[]) {
  std::string datafile;
  std::string outprefix;
  int num_points;
  int num_samples;

  int max_time;
  int max_iterations;

  po::options_description desc("Allowed options.");
  desc.add_options()
    ("datafile", po::value<std::string>(&datafile), "link to graph file")
    ("outprefix", po::value<std::string>(&outprefix)->default_value("sample"), "prefix to results files")
    ("num_points", po::value<int>(&num_points)->default_value(100), "number of points in coreset")
    ("num_samples", po::value<int>(&num_samples)->default_value(10000), "number of samples")
    ("max_time", po::value<int>(&max_time)->default_value(5), "maximum time (s) per weight solve")
    ("max_iters", po::value<int>(&max_iterations)->default_value(1000), "maximum number of iterations per weight solve");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  MatrixXd data = load_csv<MatrixXd>(datafile);
  auto mu = DataUniform(data);

  // Initial point positions
  MatrixXd points = mu(num_points);
  VectorXd weights = VectorXd::Zero(num_points);
  int num_dim = points.rows();

  // Define transport problem
  SemidiscreteTransport<DataUniform> problem(mu, points, num_samples);

  // Eigen output format
  Eigen::IOFormat csv_fmt(Eigen::FullPrecision, Eigen::DontAlignCols, ",", "\n");

#if DEBUG
  std::cout << "|Point iteration|Time|Gradient norm|Cost delta|Number iterations|" << std::endl;
  std::cout << "|-+-+-+-+-|" << std::endl;
#endif

  // AdamOptimizer
  AdamParams params;
  params.max_iterations = max_iterations;
  params.max_time = max_time;
  AdamSolver solver(params);

  // Maximum number of outer loop iterations (point updates)
  int max_iters = 10;

  double fx = 0.0;
  double fxp = fx;
  for (int iter = 1; iter <= max_iters; ++iter) {
    MatrixXd new_points = MatrixXd::Zero(num_dim, num_points);
    std::vector<double> densities(num_points, 0.0);

    auto tstart_iter = std::chrono::high_resolution_clock::now();
    int num_iters = solver.minimize(problem, weights, fx);
    auto tend_iter = std::chrono::high_resolution_clock::now();

    problem.barycenters(weights, new_points, densities);
    for (int i = 0; i < num_points; ++i) {
      if (densities[i] > 0.0) {
        new_points.col(i) /= densities[i];
      } else {
        new_points.col(i) = points.col(i);
      }
    }

#if DEBUG
    VectorXd grad(weights.size());
    problem(weights, grad);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tend_iter - tstart_iter).count();

    std::cout << "|" << iter << "|" << duration / 1000. << "s|" << grad.norm()
              << "|" << std::abs(fx - fxp) << "|" << num_iters << "|" << std::endl;
#endif
    fxp = fx;
    problem.points = new_points;
  }

#if DEBUG
    std::string filename = "./results/" + outprefix + ".csv";
    std::ofstream fout(filename, std::ofstream::out);
    fout << problem.points.transpose().format(csv_fmt) << std::endl;
    fout.close();
#endif
}
