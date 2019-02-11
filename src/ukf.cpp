#include <iostream>
#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;


double normalize_angle(double rad) {
  if (rad >= M_PI) return normalize_angle(rad - 2 * M_PI);
  else if (rad < -M_PI) return normalize_angle(rad + 2 * M_PI);
  return rad;
}

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(5);
  x_.setZero();

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 6;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.9;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (MaybeInitialize(meas_package)) {
    return;
  }
  double delta_t = (meas_package.timestamp_ - time_us_) / 1e6;
  if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    Prediction(delta_t);
    UpdateLidar(meas_package);
    time_us_ = meas_package.timestamp_;
  } else if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    Prediction(delta_t);
    UpdateRadar(meas_package);
    time_us_ = meas_package.timestamp_;
  }
}

bool UKF::MaybeInitialize(MeasurementPackage meas_package) {
  if (is_initialized_) return false;
  if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    is_initialized_ = true;
    x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    P_.setZero();
    P_(0,0) = 10;
    P_(1,1) = 10;
    P_(2,2) = 10;
    P_(3,3) = 10;
    P_(4,4) = 10;
    time_us_ = meas_package.timestamp_;
    return true;
  } else if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    is_initialized_ = true;
    x_ << meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]),
      meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]),
      0,
      0,
      0;
    P_.setZero();
    P_(0,0) = 10;
    P_(1,1) = 10;
    P_(2,2) = 10;
    P_(3,3) = 10;
    P_(4,4) = 10;
    time_us_ = meas_package.timestamp_;
    return true;
  }
  return false;
}

void UKF::Prediction(double delta_t) {
  Xsig_pred_.resize(n_x_, n_aug_ * 2 + 1);
  VectorXd x_aug(n_aug_);
  x_aug.setZero();
  x_aug.head(n_x_) = x_;

  MatrixXd P_aug(n_aug_, n_aug_);
  P_aug.setZero();
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  MatrixXd Xsig_aug(n_aug_, n_aug_ * 2 + 1);
  Xsig_aug.setZero();
  Xsig_aug.col(0) = x_aug;

  MatrixXd sig_diff(n_aug_, n_aug_);
  sig_diff = P_aug.llt().matrixL();

  Xsig_aug.middleCols(1, n_aug_) = sqrt(lambda_ + n_aug_) * sig_diff;
  Xsig_aug.middleCols(1, n_aug_).colwise() += x_aug;
  Xsig_aug.middleCols(n_aug_ + 1, n_aug_) = -sqrt(lambda_ + n_aug_) * sig_diff;
  Xsig_aug.middleCols(n_aug_ + 1, n_aug_).colwise() += x_aug;

  Xsig_pred_.setZero();
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    Xsig_pred_.col(i) = Xsig_aug.col(i).head(n_x_);
    if (yawd == 0) {
      Xsig_pred_(0, i) += v * cos(yaw) * delta_t;
      Xsig_pred_(1, i) += v * sin(yaw) * delta_t;
      Xsig_pred_(3, i) += normalize_angle(yawd * delta_t);
    } else {
      Xsig_pred_(0, i) += v * (sin(yaw + yawd * delta_t) - sin(yaw)) / yawd;
      Xsig_pred_(1, i) += v * (-cos(yaw + yawd * delta_t) + cos(yaw)) / yawd;
      Xsig_pred_(3, i) += normalize_angle(yawd * delta_t);
    }

    Xsig_pred_(0, i) += 0.5 * delta_t * delta_t * cos(yaw) * nu_a;
    Xsig_pred_(1, i) += 0.5 * delta_t * delta_t * sin(yaw) * nu_a;
    Xsig_pred_(2, i) += nu_a * delta_t;
    Xsig_pred_(3, i) += 0.5 * delta_t * delta_t * nu_yawdd;
    Xsig_pred_(4, i) += delta_t * nu_yawdd;
  }

  double w0 = lambda_ / (lambda_ + n_aug_);
  double wi = 1 / (2.0 * (lambda_ + n_aug_));
  x_ = w0 * Xsig_pred_.col(0);
  x_ += wi * (Xsig_pred_.rightCols(2 * n_aug_).rowwise().sum().array()).matrix();

  x_(3) = normalize_angle(x_(3));
  x_(4) = normalize_angle(x_(4));

  P_.setZero();
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    double w = (i == 0) ? w0 : wi;
    VectorXd xdiff = Xsig_pred_.col(i) - x_;
    xdiff(3) = normalize_angle(xdiff(3));
    xdiff(4) = normalize_angle(xdiff(4));
    P_ += w * xdiff * xdiff.transpose();
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  double w0 = lambda_ / (lambda_ + n_aug_);
  double wi = 1 / (2.0 * (lambda_ + n_aug_));
  VectorXd z_pred(2);
  MatrixXd S(2, 2);
  z_pred = w0 * Xsig_pred_.col(0).head(2);
  for (int i = 1; i < Xsig_pred_.cols(); ++i) {
    z_pred += wi * Xsig_pred_.col(i).head(2);
  }
  S.setZero();
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    VectorXd meandiff = Xsig_pred_.col(i).head(2) - z_pred;
    S += (i == 0 ? w0 : wi) * meandiff * meandiff.transpose();
  }
  S(0,0) += std_laspx_ * std_laspx_;
  S(1,1) += std_laspy_ * std_laspy_;

  MatrixXd T (n_x_, 2);
  T.setZero();
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    T += (i == 0 ? w0 : wi) * (Xsig_pred_.col(i) - x_) * (Xsig_pred_.col(i).head(2) - z_pred).transpose();
  }
  MatrixXd K = T * S.inverse();
  x_ = x_ + K * (meas_package.raw_measurements_ - z_pred);
  x_(3) = normalize_angle(x_(3));
  x_(4) = normalize_angle(x_(4));
  P_ = P_ - K * S * K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  double w0 = lambda_ / (lambda_ + n_aug_);
  double wi = 1 / (2.0 * (lambda_ + n_aug_));
  VectorXd z_pred(3);
  MatrixXd Zsig_pred(3, Xsig_pred_.cols());
  MatrixXd S(3, 3);
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    double rho = sqrt(px*px + py*py);
    Zsig_pred.col(i) << rho,
      atan2(py, px),
      (px * cos(yaw) * v + py * sin(yaw) * v) / rho;
  }

  z_pred.setZero();
  for (int i = 0; i < Zsig_pred.cols(); ++i) {
    double w = (i == 0 ? w0 : wi);
    z_pred(0) += w * Zsig_pred(0, i);
    z_pred(1) += w * Zsig_pred(1, i);
    z_pred(2) += w * Zsig_pred(2, i);
  }

  S.setZero();
  for (int i = 0; i < Zsig_pred.cols(); ++i) {
    VectorXd meandiff = Zsig_pred.col(i) - z_pred;
    S += (i == 0 ? w0 : wi) * meandiff * meandiff.transpose();
  }

  S(0,0) += std_radr_ * std_radr_;
  S(1,1) += std_radphi_ * std_radphi_;
  S(2,2) += std_radrd_ * std_radrd_;

  MatrixXd T (n_x_, 3);
  T.setZero();
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    VectorXd xdiff = Xsig_pred_.col(i) - x_;
    xdiff(3) = normalize_angle(xdiff(3));
    xdiff(4) = normalize_angle(xdiff(4));
    VectorXd zdiff = Zsig_pred.col(i) - z_pred;
    zdiff(1) = normalize_angle(zdiff(1));
    T += (i == 0 ? w0 : wi) * xdiff * zdiff.transpose();
  }
  MatrixXd K = T * S.inverse();
  VectorXd zdiff = meas_package.raw_measurements_ - z_pred;
  zdiff(1) = normalize_angle(zdiff(1));
  x_ = x_ + K * zdiff;
  x_(3) = normalize_angle(x_(3));
  x_(4) = normalize_angle(x_(4));
  P_ = P_ - K * S * K.transpose();
}
