
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h> 
#include  <pybind11/chrono.h>
#include <cmath>
#include <vector>

namespace py = pybind11;

// Helper function to check if a value is within a range
bool in_range(double value, double min, double max) {
    return (value >= min) && (value <= max);
}

std::tuple<bool, double, double> ray_intersects_segment(double x, double y, double angle, double x1, double y1, double x2, double y2) {
    double dx = std::cos(angle);
    double dy = std::sin(angle);

    double denominator = (y2 - y1) * (dx) - (x2 - x1) * (dy);
    if (denominator == 0) {
        return std::make_tuple(false, 0.0, 0.0); // Lines are parallel
    }

    double ua = ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) / denominator;
    double ub = ((dx) * (y - y1) - (dy) * (x - x1)) / denominator;

    if (ua >= 0 && in_range(ub, 0, 1)) {
        double ix = x + ua * (dx);
        double iy = y + ua * (dy);
        return std::make_tuple(true, ix, iy);
    }
    return std::make_tuple(false, 0.0, 0.0);
}

std::tuple<double, int> get_sensor_data_single(
    const std::vector<std::tuple<double, double, double, double>>& edges_list,
    double sensor_range,
    double car_theta,
    double car_x,
    double car_y,
    double angle,
    const std::vector<std::vector<double>>& circle_obs) {

    double sensor_angle = car_theta + angle;
    double min_distance = sensor_range;
    int contact_edge = -1;
    int edlen = edges_list.size();

    for (int edge_ind = 0; edge_ind < edlen; ++edge_ind) {
        double x1, y1, x2, y2;
        std::tie(x1, y1, x2, y2) = edges_list[edge_ind];

        bool intersection;
        double ix, iy;
        std::tie(intersection, ix, iy) = ray_intersects_segment(car_x, car_y, sensor_angle, x1, y1, x2, y2);

        if (intersection) {
            double distance = std::sqrt(std::pow(ix - car_x, 2) + std::pow(iy - car_y, 2));
            if (distance < min_distance) {
                min_distance = distance;
                contact_edge = edge_ind;
            }
        }
    }

    for (const auto& circle : circle_obs) {
        // laser ray intersects with circle
        double b = std::sqrt(std::pow(car_x - circle[0], 2) + std::pow(car_y - circle[1], 2));
        double angle_to_circle = std::atan2(circle[1] - car_y, circle[0] - car_x);
        double angle_diff = std::fmod(angle_to_circle - sensor_angle + M_PI, 2 * M_PI) - M_PI;
        if (std::abs(angle_diff) < std::asin(circle[2] / b)) {
            double distance = std::sqrt(std::pow(circle[2], 2) + std::pow(b, 2) - 2 * circle[2] * b * std::cos(std::abs(angle_diff)));
            if (distance < min_distance) {
                min_distance = distance;
            }
        }
    }

    return std::make_tuple(min_distance, contact_edge);
}


py::array_t<double> get_sensor_data1(
    int egoid,
    const std::vector<std::tuple<double, double, double, double>>& edges_list,
    int num_sensors,
    double sensor_range,
    double car_theta,
    double car_x,
    double car_y,
    const std::vector<std::vector<double>>& env_circle_obs) {

    std::vector<std::vector<double>> circle_obs1;
    for (const auto& circle : env_circle_obs) {
        if (circle[0] == car_x && circle[1] == car_y) {
            continue;
        }
        if (std::sqrt(std::pow(circle[0] - car_x, 2) + std::pow(circle[1] - car_y, 2)) > circle[2] + sensor_range) {
            continue;
        } else {
            circle_obs1.push_back({circle[0], circle[1], circle[2]});
        }
    }

    py::array_t<double> sensor_data(num_sensors);
    double* sensor_data_ptr = sensor_data.mutable_data();
    for (int i = 0; i < num_sensors; ++i) {
        sensor_data_ptr[i] = sensor_range;
    }

    std::vector<double> angles(num_sensors);
    for (int i = 0; i < num_sensors; ++i) {
        angles[i] = -M_PI / 2 + (M_PI * i) / (num_sensors -1.0);
    }

    for (int i = 0; i < num_sensors; ++i) {
        double distance;
        int contact_edge;
        std::tie(distance, contact_edge) = get_sensor_data_single(edges_list, sensor_range, car_theta, car_x, car_y, angles[i], circle_obs1);
        sensor_data_ptr[i] = distance;
    }

    return sensor_data;
}

bool check_car_collision(const std::vector<double>& new_car, const std::vector<std::vector<double>>& cars, double safety_distance) {
    // theoretically, this function should first check two points can see-through, then check the distance
    for (const auto& car : cars) {
        double distance = std::sqrt(std::pow(new_car[0] - car[0], 2) + std::pow(new_car[1] - car[1], 2));
        if (distance < safety_distance) {
            return true;
        }
    }
    return false;
}

PYBIND11_MODULE(get_sensor_data, m) {
    m.def("ray_intersects_segment", &ray_intersects_segment, "Ray-segment intersection check");
    m.def("get_sensor_data_single", &get_sensor_data_single, "Get single sensor data");
    m.def("get_sensor_data1", &get_sensor_data1, "Get sensor data");
    m.def("check_car_collision", &check_car_collision, "Check car collision");
}