#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "radar_utils.hpp"

static inline bool exists(const std::string& name) {
    struct stat buffer;
    return !(stat (name.c_str(), &buffer) == 0);
}

// assumes file names are EPOCH times which can be sorted numerically
struct less_than_img {
    inline bool operator() (const std::string& img1, const std::string& img2) {
        std::vector<std::string> parts;
        boost::split(parts, img1, boost::is_any_of("."));
        int64 i1 = std::stoll(parts[0]);
        boost::split(parts, img2, boost::is_any_of("."));
        int64 i2 = std::stoll(parts[0]);
        return i1 < i2;
    }
};

void get_file_names(std::string path, std::vector<std::string> &files, std::string extension) {
    DIR *dirp = opendir(path.c_str());
    std::cout<<path<<std::endl;
    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL) {
        if (exists(dp->d_name)) {
            if (!extension.empty()) {
                std::vector<std::string> parts;
                boost::split(parts, dp->d_name, boost::is_any_of("."));
                if (parts[parts.size() - 1].compare(extension) != 0)
                    continue;
            }
            files.push_back(dp->d_name);
        }
    }
    // Sort files in ascending order of time stamp
    std::sort(files.begin(), files.end(), less_than_img());
}



void load_radar(std::string path, std::vector<int64_t> &timestamps, std::vector<double> &azimuths,
    std::vector<bool> &valid, cv::Mat &fft_data, int navtech_version) {
    int encoder_size = 5600;
    
    cv::Mat raw_example_data = cv::imread(path, cv::IMREAD_GRAYSCALE);
    int N = raw_example_data.rows;
    timestamps = std::vector<int64_t>(N, 0);
    azimuths = std::vector<double>(N, 0);
    valid = std::vector<bool>(N, true);
    int range_bins = 3768;
    if (navtech_version == CIR204)
        range_bins = 3360;
    fft_data = cv::Mat::zeros(N, range_bins, CV_32F);
#pragma omp parallel
    for (int i = 0; i < N; ++i) {
        uchar* byteArray = raw_example_data.ptr<uchar>(i);
        timestamps[i] = *((int64_t *)(byteArray));
        azimuths[i] = *((uint16_t *)(byteArray + 8)) * 2 * M_PI / double(encoder_size);
        valid[i] = byteArray[10] == 255;
        // std::cout<<timestamps[i]<<","<<azimuths[i]*5600/2/M_PI<<std::endl;
        for (int j = 42; j < range_bins; j++) {
            fft_data.at<float>(i, j) = (float)*(byteArray + 11 + j) / 255.0;
            // std::cout<<fft_data.at<float>(i, j)<<std::endl;
            // cv::waitKey();
        }
    }
    // cv::imshow("fft",fft_data);
    // cv::waitKey();
    // std::cout<<fft_data.size()<<std::endl;
}

// Mulran
void load_radar2(std::string path, std::vector<int64_t> &timestamps, std::vector<double> &azimuths,
    std::vector<bool> &valid, cv::Mat &fft_data, int navtech_version) {
    int encoder_size = 5600;
    int N = 400;
    timestamps = std::vector<int64_t>(N, 0);
    azimuths = std::vector<double>(N, 0);
    valid = std::vector<bool>(N, true);
    int range_bins = 3768;
    if (navtech_version == CIR204)
        range_bins = 3360;
    fft_data = cv::Mat::zeros(N, range_bins, CV_32F);
    
    const char delimiter = ',';
    std::ifstream file(path);
    std::string line;
    int rows = 0;
    int cols = 0;
    if (file.is_open()) {
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            int current_cols = 0;
            while (std::getline(ss, cell, delimiter)) {
                double value = std::stof(cell);
                if(current_cols==0){
                    timestamps[rows] = value;
                    current_cols++;
                    continue;
                }
                else if(current_cols==1 || current_cols==3){
                    current_cols++;
                    continue;
                }
                else if(current_cols==2){
                    azimuths[rows] = value * 2 * M_PI / double(encoder_size);
                    current_cols++;
                    continue;
                }
                // std::cout<<value<<std::endl;
                fft_data.at<float>(rows, current_cols-4) = (float)value/255.0;
                current_cols++;
            }
            if (rows == 0) {
                cols = current_cols;
            } else if (cols != current_cols) {
                std::cerr << "Inconsistent column count in CSV file" << std::endl;
                exit(1);
            }
            ++rows;
        }
    } else {
        std::cerr << "Unable to open CSV file" << std::endl;
        exit(1);
    }
    // std::cout<<fft_data.size()<<std::endl;
    // cv::imshow("fft",fft_data);
    // cv::waitKey();
    
}

// file is in the oxford dataset format
void load_velodyne(std::string path, std::vector<int64_t> &timestamps, std::vector<double> &azimuths,
    Eigen::MatrixXd &pc) {
    double hdl32e_range_resolution = 0.002;
    cv::Mat data = cv::imread(path, cv::IMREAD_GRAYSCALE);
    uint N = data.cols;
    cv::transpose(data, data);
    timestamps = std::vector<int64_t>(N * 32, 0);
    azimuths = std::vector<double>(N * 32, 0.0);
    Eigen::MatrixXd ranges = Eigen::MatrixXd::Zero(N, 32);
    for (uint i = 0; i < N; ++i) {
        uchar* byteArray = data.ptr<uchar>(i);
        uint16_t azimuth = *( (uint16_t *)(byteArray + 96));
        double a = double(azimuth) * M_PI / 18000.0;
        int64_t t = *((int64_t *)(byteArray + 98));
        int k = 0;
        for (uint j = 32; j < 96; j += 2) {
            uint16_t range = *( (uint16_t *)(byteArray + j));
            ranges(i, k) = double(range) * hdl32e_range_resolution;
            azimuths[i * 32 + k] = a;
            timestamps[i * 32 + k] = t;
            k++;
        }
    }
    // Convert to 3D point cloud
    std::vector<double> elevations = {-0.1862, -0.1628, -0.1396, -0.1164, -0.0930, -0.0698, -0.0466, -0.0232, 0.,
        0.0232, 0.0466, 0.0698, 0.0930, 0.1164, 0.1396, 0.1628, 0.1862, 0.2094, 0.2327, 0.2560, 0.2793, 0.3025, 0.3259,
        0.3491, 0.3723, 0.3957, 0.4189, 0.4421, 0.4655, 0.4887, 0.5119, 0.5353};
    double hdl32e_base_to_fire_height = 0.090805;
    // double hdl32e_minimum_range = 1.0;
    std::vector<double> x, y, z;
    for (uint i = 0; i < N; ++i) {
        for (uint j = 0; j < 32; ++j) {
            z.push_back(sin(elevations[j]) * ranges(i, j) - hdl32e_base_to_fire_height);
            double xy = cos(elevations[j]) * ranges(i, j);
            x.push_back(sin(azimuths[i * 32 + j]) * xy);
            y.push_back(-cos(azimuths[i * 32 + j]) * xy);
        }
    }
    pc = Eigen::MatrixXd::Zero(3, z.size());
    for (uint i = 0; i < z.size(); ++i) {
        pc(0, i) = x[i];
        pc(1, i) = y[i];
        pc(2, i) = z[i];
    }
}

// file is the result of saving a D X N pointcloud into a .txt file (comma-separated)
void load_velodyne2(std::string path, Eigen::MatrixXd &pc) {
    std::ifstream ifs(path);
    std::string line;
    int N = 0;
    while (std::getline(ifs, line))
        ++N;
    ifs = std::ifstream(path);
    pc = Eigen::MatrixXd::Zero(3, N);
    int i = 0;
    while (std::getline(ifs, line)) {
        std::vector<std::string> parts;
        boost::split(parts, line, boost::is_any_of(","));
        for (int j = 0; j < 3; ++j) {
            pc(j, i) = std::stod(parts[j]);
        }
        i++;
    }
}

static float getFloatFromByteArray(char *byteArray, uint index) {
    return *( (float *)(byteArray + index));
}

// Input is a .bin binary file.
void load_velodyne3(std::string path, Eigen::MatrixXd &pc, Eigen::MatrixXd & intensities, std::vector<float> &times) {
    std::ifstream ifs(path, std::ios::binary);
    std::vector<char> buffer(std::istreambuf_iterator<char>(ifs), {});
    int float_offset = 4;
    int fields = 6;  // x, y, z, i, r, t
    int N = buffer.size() / (float_offset * fields);
    int point_step = float_offset * fields;
    pc = Eigen::MatrixXd::Ones(4, N);
    intensities = Eigen::MatrixXd::Zero(1, N);
    times = std::vector<float>(N);
    int j = 0;
    for (uint i = 0; i < buffer.size(); i += point_step) {
        pc(0, j) = getFloatFromByteArray(buffer.data(), i);
        pc(1, j) = getFloatFromByteArray(buffer.data(), i + float_offset);
        pc(2, j) = getFloatFromByteArray(buffer.data(), i + float_offset * 2);
        intensities(0, j) = getFloatFromByteArray(buffer.data(), i + float_offset * 3);
        times[j] = getFloatFromByteArray(buffer.data(), i + float_offset * 5);
        j++;
    }
}

double get_azimuth_index(std::vector<double> &azimuths, double azimuth) {
    double mind = 1000;
    double closest = 0;
    int M = azimuths.size();
    for (uint i = 0; i < azimuths.size(); ++i) {
        double d = fabs(azimuths[i] - azimuth);
        if (d < mind) {
            mind = d;
            closest = i;
        }
    }
    if (azimuths[closest] < azimuth) {
        double delta = 0;
        if (closest < M - 1)
            delta = (azimuth - azimuths[closest]) / (azimuths[closest + 1] - azimuths[closest]);
        closest += delta;
    } else if (azimuths[closest] > azimuth){
        double delta = 0;
        if (closest > 0)
            delta = (azimuths[closest] - azimuth) / (azimuths[closest] - azimuths[closest - 1]);
        closest -= delta;
    }
    return closest;
}

void radar_polar_to_cartesian(std::vector<double> &azimuths, cv::Mat &fft_data, float radar_resolution,
    float cart_resolution, int cart_pixel_width, bool interpolate_crossover, cv::Mat &cart_img, int output_type,
    int navtech_version) {

    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;

    cv::Mat map_x = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);
    cv::Mat map_y = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);

#pragma omp parallel for collapse(2)
    for (int j = 0; j < map_y.cols; ++j) {
        for (int i = 0; i < map_y.rows; ++i) {
            map_y.at<float>(i, j) = -1 * cart_min_range + j * cart_resolution;
        }
    }
#pragma omp parallel for collapse(2)
    for (int i = 0; i < map_x.rows; ++i) {
        for (int j = 0; j < map_x.cols; ++j) {
            map_x.at<float>(i, j) = cart_min_range - i * cart_resolution;
        }
    }
    cv::Mat range = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);
    cv::Mat angle = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);

    double azimuth_step = azimuths[1] - azimuths[0];
#pragma omp parallel for collapse(2)
    for (int i = 0; i < range.rows; ++i) {
        for (int j = 0; j < range.cols; ++j) {
            float x = map_x.at<float>(i, j);
            float y = map_y.at<float>(i, j);
            float r = (sqrt(pow(x, 2) + pow(y, 2)) - radar_resolution / 2) / radar_resolution;
            if (r < 0)
                r = 0;
            range.at<float>(i, j) = r;
            float theta = atan2f(y, x);
            if (theta < 0)
                theta += 2 * M_PI;
            if (navtech_version == CIR204) {
                angle.at<float>(i, j) = get_azimuth_index(azimuths, theta);
            } else {
                angle.at<float>(i, j) = (theta - azimuths[0]) / azimuth_step;
            }
        }
    }
    if (interpolate_crossover) {
        cv::Mat a0 = cv::Mat::zeros(1, fft_data.cols, CV_32F);
        cv::Mat aN_1 = cv::Mat::zeros(1, fft_data.cols, CV_32F);
        for (int j = 0; j < fft_data.cols; ++j) {
            a0.at<float>(0, j) = fft_data.at<float>(0, j);
            aN_1.at<float>(0, j) = fft_data.at<float>(fft_data.rows-1, j);
        }
        cv::vconcat(aN_1, fft_data, fft_data);
        cv::vconcat(fft_data, a0, fft_data);
        angle = angle + 1;
    }
    cv::remap(fft_data, cart_img, range, angle, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    if (output_type == CV_8UC1) {
        double min, max;
        cv::minMaxLoc(cart_img, &min, &max);
        cart_img.convertTo(cart_img, CV_8UC1, 255.0 / max);
    }
}

void polar_to_cartesian_points(std::vector<double> azimuths, Eigen::MatrixXd polar_points,
    float radar_resolution, Eigen::MatrixXd &cart_points) {
    cart_points = polar_points;
    for (uint i = 0; i < polar_points.cols(); ++i) {
        double azimuth = azimuths[polar_points(0, i)];
        double r = polar_points(1, i) * radar_resolution + radar_resolution / 2;
        cart_points(0, i) = r * cos(azimuth);
        cart_points(1, i) = r * sin(azimuth);
    }
}

void polar_to_cartesian_points(std::vector<double> azimuths, std::vector<int64_t> times, Eigen::MatrixXd polar_points,
    float radar_resolution, Eigen::MatrixXd &cart_points, std::vector<int64_t> &point_times) {
    cart_points = polar_points;
    point_times = std::vector<int64_t>(polar_points.cols());
    for (uint i = 0; i < polar_points.cols(); ++i) {
        double azimuth = azimuths[polar_points(0, i)];
        double r = polar_points(1, i) * radar_resolution + radar_resolution / 2;
        cart_points(0, i) = r * cos(azimuth);
        cart_points(1, i) = r * sin(azimuth);
        point_times[i] = times[polar_points(0, i)];
    }
}

void convert_to_bev(Eigen::MatrixXd &cart_points, float cart_resolution, int cart_pixel_width,
    std::vector<cv::Point2f> &bev_points) {
    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;
    bev_points.clear();
    int j = 0;
    for (uint i = 0; i < cart_points.cols(); ++i) {
        double u = (cart_min_range + cart_points(1, i)) / cart_resolution;
        double v = (cart_min_range - cart_points(0, i)) / cart_resolution;
        if (0 < u && u < cart_pixel_width && 0 < v && v < cart_pixel_width) {
            bev_points.push_back(cv::Point2f(u, v));
            cart_points(0, j) = cart_points(0, i);
            cart_points(1, j) = cart_points(1, i);
            j++;
        }
    }
    cart_points.conservativeResize(3, bev_points.size());
}

void convert_to_bev(Eigen::MatrixXd &cart_points, float cart_resolution, int cart_pixel_width, int patch_size,
    std::vector<cv::KeyPoint> &bev_points, std::vector<int64_t> &point_times) {
    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;
    bev_points.clear();
    int j = 0;
    for (uint i = 0; i < cart_points.cols(); ++i) {
        double u = (cart_min_range + cart_points(1, i)) / cart_resolution;
        double v = (cart_min_range - cart_points(0, i)) / cart_resolution;
        if (0 < u - patch_size && u + patch_size < cart_pixel_width && 0 < v - patch_size &&
            v + patch_size < cart_pixel_width) {
            bev_points.push_back(cv::KeyPoint(u, v, patch_size));
            point_times[j] = point_times[i];
            cart_points(0, j) = cart_points(0, i);
            cart_points(1, j) = cart_points(1, i);
            j++;
        }
    }
    point_times.resize(bev_points.size());
    cart_points.conservativeResize(3, bev_points.size());
}

void convert_from_bev(std::vector<cv::KeyPoint> bev_points, float cart_resolution, int cart_pixel_width,
    Eigen::MatrixXd &cart_points) {
    cart_points = Eigen::MatrixXd::Zero(2, bev_points.size());
    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;
    for (uint i = 0; i < bev_points.size(); ++i) {
        cart_points(0, i) = cart_min_range - cart_resolution * bev_points[i].pt.y;
        cart_points(1, i) = cart_resolution * bev_points[i].pt.x - cart_min_range;
    }
}

void draw_points(cv::Mat cart_img, Eigen::MatrixXd cart_targets, float cart_resolution, int cart_pixel_width,
    cv::Mat &vis, std::vector<uint> color) {
    std::vector<cv::Point2f> bev_points;
    convert_to_bev(cart_targets, cart_resolution, cart_pixel_width, bev_points);
    cv::cvtColor(cart_img, vis, cv::COLOR_GRAY2BGR);
    for (cv::Point2f p : bev_points) {
        // cv::circle(vis, p, 2, cv::Scalar(0, 0, 255), -1);
        if (cart_img.depth() == CV_8UC1)
            vis.at<cv::Vec3b>(int(p.y), int(p.x)) = cv::Vec3b(color[0], color[1], color[2]);
        if (cart_img.depth() == CV_32F)
            vis.at<cv::Vec3f>(int(p.y), int(p.x)) = cv::Vec3f(color[0], color[1], color[2]);
    }
}

void draw_points(cv::Mat &vis, Eigen::MatrixXd cart_targets, float cart_resolution, int cart_pixel_width,
    std::vector<uint> color) {
    std::vector<cv::Point2f> bev_points;
    convert_to_bev(cart_targets, cart_resolution, cart_pixel_width, bev_points);
    for (cv::Point2f p : bev_points) {
        cv::circle(vis, p, 1, cv::Scalar(color[0], color[1], color[2]), -1);
        // if (vis.depth() == CV_8UC1)
        //     vis.at<cv::Vec3b>(int(p.y), int(p.x)) = cv::Vec3b(color[0], color[1], color[2]);
        // if (vis.depth() == CV_32F)
        //     vis.at<cv::Vec3f>(int(p.y), int(p.x)) = cv::Vec3f(color[0], color[1], color[2]);
    }
}

// Oxford format
bool get_groundtruth_odometry(std::string gtfile, int64 t1, int64 t2, std::vector<float> &gt) {
    std::ifstream ifs(gtfile);
    std::string line;
    std::getline(ifs, line);
    gt.clear();
    bool gtfound = false;
    while (std::getline(ifs, line)) {
        std::vector<std::string> parts;
        boost::split(parts, line, boost::is_any_of(","));
        if (std::stoll(parts[9]) == t1 && std::stoll(parts[8]) == t2) {
            for (int i = 2; i < 8; ++i) {
                gt.push_back(std::stof(parts[i]));
            }
            gtfound = true;
            break;
        }
    }
    return gtfound;
}

// Boreas format
bool get_groundtruth_odometry2(std::string gtfile, int64_t t, std::vector<double> &gt) {
    std::ifstream ifs(gtfile);
    std::string line;
    gt.clear();
    bool gtfound = false;
    double min_delta = 0.1;
    while (std::getline(ifs, line)) {
        std::vector<std::string> parts;
        boost::split(parts, line, boost::is_any_of(","));
        int64_t t2 = std::stoll(parts[0]);
        double delta = fabs((t - t2) / 1000000000.0);
        if (delta < min_delta) {
            for (uint i = 0; i < parts.size(); ++i) {
                gt.push_back(std::stod(parts[i]));
            }
            gtfound = true;
            min_delta = delta;
        }
    }
    return gtfound;
}

bool get_groundtruth_odometry3(std::string gtfile, int64 t1, std::vector<double> &gt) {
    std::ifstream ifs(gtfile);
    std::string line;
    std::getline(ifs, line);
    gt.clear();
    bool gtfound = false;
    while (std::getline(ifs, line)) {
        std::vector<std::string> parts;
        boost::split(parts, line, boost::is_any_of(","));
        if (std::stoll(parts[0]) == t1) {
            for (int i = 1; i < 3; ++i) {
                gt.push_back(std::stof(parts[i]));
            }
            gtfound = true;
            break;
        }
    }
    return gtfound;
}

// use img2
void draw_matches(cv::Mat &img, std::vector<cv::KeyPoint> kp1, std::vector<cv::KeyPoint> kp2,
    std::vector<cv::DMatch> matches, int radius) {

    for (uint i = 0; i < matches.size(); ++i) {
        cv::KeyPoint p1 = kp1[matches[i].queryIdx];
        cv::KeyPoint p2 = kp2[matches[i].trainIdx];
        cv::line(img, p2.pt, p1.pt, cv::Scalar(255, 255, 255), 2);
        cv::circle(img, p2.pt, radius, cv::Scalar(0, 0, 255), -1);
        cv::circle(img, p1.pt, radius, cv::Scalar(0, 255, 0), -1);
    }
}

static double wrapto2pi(double theta) {
    if (theta < 0) {
        return theta + 2 * M_PI;
    } else if (theta > 2 * M_PI) {
        return theta - 2 * M_PI;
    } else {
        return theta;
    }
}

void getTimes(Eigen::MatrixXd cart_targets, std::vector<double> azimuths, std::vector<int64_t> times,
    std::vector<int64_t> &tout) {
    tout.clear();
    for (uint j = 0; j < cart_targets.cols(); ++j) {
        double theta = wrapto2pi(atan2(cart_targets(1, j), cart_targets(0, j)));
        double closest = 0;
        double mindiff = 1000;
        for (uint k = 0; k < mindiff; ++k) {
            if (fabs(theta - azimuths[k]) < mindiff) {
                mindiff = fabs(theta - azimuths[k]);
                closest = k;
            }
            tout.push_back(times[closest]);
        }
    }
}

static void usage(const char *argv[]) {
    std::cerr << std::endl << std::endl;
    std::cerr << "USAGE:" << std::endl;
    std::cerr << "  " << argv[0] << " --root ROOT_DIRECTORY --sequence SEQUENCE --append APPEND_EXT" << std::endl;
    std::cerr << std::endl;
    std::cerr << "--root ROOT_DIRECTORY  Absolute path to data, ex: /home/keenan/Documents/data/" << std::endl;
    std::cerr << "--sequence SEQUENCE  Name of Oxford Sequence, ex: 2019-01-10-11-46-21-radar-oxford-10k" << std::endl;
    std::cerr << "--append APPEND_EXT   accuracy<APPEND_EXT>.csv output file name for odometry" << std::endl;
    std::cerr << std::endl;
}

int validateArgs(const int argc, const char *argv[], std::string &root) {
    std::string seq, app;
    return validateArgs(argc, argv, root, seq, app);
}

int validateArgs(const int argc, const char *argv[], std::string &root, std::string &seq, std::string &app) {
    if (argc <= 2) {
        std::cerr << "Not enough arguments, usage:";
        usage(argv);
        return 1;
    }
    for (int i = 1; i <= argc - 2; i += 2) {
        const std::string opt(argv[i]);
        if (opt == "--root") {
            root = argv[i + 1];
        } else if (opt == "--sequence") {
            seq = argv[i + 1];
        } else if (opt == "--append") {
            app = argv[i + 1];
        } else {
            std::cerr << "Unknown option " << opt << ", usage:"; usage(argv); exit(1);
        }
    }
    return 0;
}

cv::Mat noise_removal(std::vector<double> &azimuths, cv::Mat &fft_data,float multipath_thres, float radar_resolution,
                    float cart_resolution, int cart_pixel_width, bool interpolate_crossover, int navtech_version, int dataset_type, int angle1, int angle2, int recon_dist){

    // double thres = 0.1;
    // double multipath_thres = 0.3; // 0.2?
    cv::Mat polar_removed = fft_data.clone();
    
    cv::Mat img_cart_original;
    cv::Mat img_canny;
    int range_bin = fft_data.cols;
    int azimuth_bin = fft_data.rows;

    radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width, interpolate_crossover, img_cart_original, CV_8UC1, navtech_version);
    cv::imshow("cart",img_cart_original);
#pragma omp parallel for collapse(2)
    for(int az1 = 0; az1<azimuth_bin; az1++){
        for(int r1=0; r1<range_bin; r1++){

            if(fft_data.at<float>(az1,r1) < multipath_thres){
                continue;
            }
            
            for(int az2 = 0; az2<azimuth_bin; az2++){
                if(abs(az1 - az2) < angle1 || abs(az1 - az2) > angle2)
                    continue;

                for(int r2 = r1; r2<range_bin;r2++){
                    if(fft_data.at<float>(az2,r2) < multipath_thres)
                        continue;
                    
                    double angle_diff = abs(az1-az2)*2*M_PI/400;
                    double cos_angle_diff = cos(angle_diff);
                    int dist = cvRound(sqrt(r2*r2+r1*r1-2*r1*r2*cos_angle_diff));

                    if(dist>recon_dist && r1+dist < range_bin){
                        polar_removed.at<float>(az1,r1+dist)=0;
                    }
                }

            }
        }
    }


    cv::Mat img_cart_thres;

    cv::threshold(img_cart_original, img_cart_thres, 100, 255, cv::THRESH_TOZERO);

    cv::Canny(img_cart_thres, img_canny, 220, 250); // need to modify threshold 

    std::vector<cv::Vec2f> lines;

    cv::HoughLines(img_canny, lines, 1, CV_PI / 180, 100);

    cv::Mat img_hough;
    img_hough = img_cart_thres.clone();

    float img_center = cart_pixel_width/2;


    // cv::Mat img_rm1;
    // radar_polar_to_cartesian(azimuths, polar_removed, radar_resolution, cart_resolution, cart_pixel_width, interpolate_crossover, img_rm1, CV_8UC1);
    // cv::imshow("img_rm_bf",img_rm1);


    for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
        

		cv::Point pt1, pt2;
		double cos_t = cos(theta), sin_t = sin(theta);
		double x0 = cos_t * rho, y0 = sin_t * rho;

		pt1.x = cvRound(x0 + 1000 * (-sin_t));
		pt1.y = cvRound(y0 + 1000 * (cos_t));
		pt2.x = cvRound(x0 - 1000 * (-sin_t));
		pt2.y = cvRound(y0 - 1000 * (cos_t));

		// cv::line(img_cart_original, pt1, pt2, cv::Scalar(255,0,255), 2, 8);
        
        float center_dist = fabs(cos_t * img_center + sin_t * img_center - rho);

        if(center_dist>30){
            // cv::line(img_cart_original, pt1, pt2, cv::Scalar(255,0,255), 2, 8);
            // std::cout<<rho<<", "<<theta*180/M_PI<<std::endl;
#pragma omp parallel for collapse(2)
            for(int t = 0; t<azimuth_bin; t++){
                for(int r=0; r<range_bin; r++){
                    float rr = r * cart_pixel_width/2 / range_bin;
                    // std::cout<<rr<<std::endl;
                    int x = img_center + cvRound( rr * sin(t*2*M_PI/azimuth_bin));
                    int y = img_center - cvRound( rr * cos(t*2*M_PI/azimuth_bin));

                    float dist = fabs(cos_t * x + sin_t * y - rho);
                    // std::cout<<cv::Point(x,y)<<"dist: "<<dist<<std::endl;
                    if((r<range_bin/2 && x>=0 && x < cart_pixel_width && y>=0 && y < cart_pixel_width && dist<40) || fft_data.at<float>(t,r) < multipath_thres){                            
                        // cv::circle(img_cart_original, cv::Point(x,y), 3,(255,0,255),1,8);
                        polar_removed.at<float>(t,r) = fft_data.at<float>(t,r);
                    }
                    
                }
            }

//             int rho_range=5;
//             for(int distance = rho - rho_range; distance < rho + rho_range; distance++){
// #pragma omp parallel for
//                 for (int x = 0; x < cart_pixel_width; x++) {
//                     int y = cvRound((distance - x * cos_t) / sin_t);
//                     if (y >= 0 && y < cart_pixel_width) {
                        
//                         cv::circle(img_cart_original, cv::Point(x,y), 3,(255,0,255),1,8);
//                         float r = (sqrt(pow(x, 2) + pow(y, 2)) - radar_resolution / 2) / radar_resolution;
//                         float t = atan2f(y, x);
//                         if (t < 0)
//                             t += 2 * M_PI;

//                         if(r < range_bin && t < azimuth_bin){
//                             polar_removed.at<float>(t,r) = fft_data.at<float>(t,r);
//                         }
//                     }
//                 }
//             }
        }

	}
    
    // visualization
    cv::Mat img_rm;
    radar_polar_to_cartesian(azimuths, polar_removed, radar_resolution, cart_resolution, cart_pixel_width, interpolate_crossover, img_rm, CV_8UC1, navtech_version);
    
    // cv::imshow("img_rm1",img_rm);
    // cv::waitKey();
    // // cv::imshow("img_original",img_cart_original);
    // // // cv::imshow("img_canny",img_canny);
    // // // cv::imshow("img_hough",img_hough);
    
    
    return polar_removed;
}