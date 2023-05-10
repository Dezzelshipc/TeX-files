#include <iostream>
#include <cmath>
#include <vector>

double f(double x, double y) {
    return sqrt(std::pow(std::cos(1 - y - x), -2) + std::tan(x * y) - 2);
}

int main() {
    double x_0 = M_PI / 4,
            x_n = M_PI / 3,
            y_prev = 1;
    int n = 10000;
    double h = (x_n - x_0) / n;

    std::vector<double> x;
    for (int i = 0; i < n; ++i) {
        x.push_back(x_0 + i * h);
    }

    std::vector<double> y1{y_prev};
    std::vector<double> y2{y_prev};

    for (int i = 0; i < n - 1; ++i) {
        y1.push_back(y1[i] + h * f(x[i], y1[i]));
        y2.push_back(y2[i] - h * f(x[i], y2[i]));
    }

    std::cout << y1.back() << ' ' << y2.back();
}