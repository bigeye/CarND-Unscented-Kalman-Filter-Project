#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd ret(4);
  ret.setZero();
  for (int i = 0; i < (int) estimations.size(); ++i) {
    ret += (estimations[i] - ground_truth[i]).array().square().matrix();
  }
  ret = ret.array() / (double) estimations.size();
  return ret;
}
