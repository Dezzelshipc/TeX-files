#include <iostream>
#include <cmath>
#include <vector>

double f(double x, double y) {
    return (std::acos(std::exp(x - y)) 
        + std::pow(std::tan(1 / std::cos(x * y)), 2) 
        + std::tan(y)) / std::sin(x);
}

int main() {
    double x_0 = M_PI / 3,
            x_n = 1,
            y_prev = std::log(7);
    int n = 10000;
    double h = (x_n - x_0) / n;

    std::vector<double> x;
    for (int i = 0; i < n; ++i) {
        x.push_back(x_0 + i * h);
    }

    std::vector<double> y{y_prev};

    for (int i = 0; i < n - 1; ++i) {
        y.push_back(y[i] + h * f(x[i], y[i]));
    }

    std::cout << y.back();
}