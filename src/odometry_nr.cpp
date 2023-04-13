#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "radar_utils.hpp"
#include "features.hpp"
#include "association.hpp"

int main(int argc, const char *argv[]) {

    // check radar dataset type
    int dataset_type = 1; // 0: Oxford, 1: Mulran

    std::string root, sequence, append;
    validateArgs(argc, argv, root, sequence, append);
    std::cout <<"seq: "<< sequence << std::endl;
    
    omp_set_num_threads(8);
    std::string datadir = root + sequence + "/radar";
    // Oxford original
    // std::string gt = root + sequence + "/gt/radar_odometry.csv";

    // Oxford mine
    std::string gt = root + sequence + "/gt_radar.csv";

    // std::string gt = root + sequcne + "/gps/gps.csv"
    // Mulran
    if(dataset_type==1){
        datadir = root + sequence + "/sensor_data/radar/ray";
        gt = root + sequence + "/gt_radar.csv";
    }
    

    // Oxford
    int min_range = 58;                 // min range of radar points (bin)
    float radar_resolution = 0.0432;    // resolution of radar bins in meters per bin
    // float cart_resolution = 0.2592;     // meters per pixel 0.0432 * 6
    float cart_resolution = 0.0432*8;
    int cart_pixel_width = 942;         // height and width of cartesian image in pixels 3856 + 12 = 3868 original code: 964 3768/4 = 942
    bool interp = true;
    int navtech_version = CTS350;

    
    //Mulran
    if(dataset_type==1){
        min_range = 58;                 // min range of radar points (bin)
        radar_resolution = 0.06;    // resolution of radar bins in meters per bin
        cart_resolution = 0.06*8;
        cart_pixel_width = 840;         // height and width of cartesian image in pixels 3360/4 = 840
        interp = true;
        navtech_version = CIR204;
    }

    int keypoint_extraction = 0;        // 0: cen2018, 1: cen2019, 2: orb

    // cen2018 parameters
    float zq = 2.0;
    int sigma_gauss = 17;
    float multipath_thres = 0.3;
    int ang1 = 60;
    int ang2 = 120;
    int recon_dist = 100;
   
    // cen2019 parameters
    int max_points = 10000;
    // ORB descriptor / matching parameters
    int patch_size = 21;                // width of patch in pixels in cartesian radar image
    float nndr = 0.70;                  // Nearest neighbor distance ratio

    // RANSAC
    double ransac_threshold = 0.2;
    double inlier_ratio = 0.90;
    int max_iterations = 100;
    // MDRANSAC
    int max_gn_iterations = 1000;
    double md_threshold = 0.2; //pow(ransac_threshold, 2);
    // DOPPLER
    double beta = 0.049;

    if(dataset_type==1){
        zq = 2.0;
        sigma_gauss = 17;
        multipath_thres = 0.3;
        ang1 = 60;
        ang2 = 120;
        recon_dist = 100;
        nndr = 0.60;
        ransac_threshold = 0.15;
    }

    // Get file names of the radar images
    std::vector<std::string> radar_files;
    get_file_names(datadir, radar_files);
    // File for storing the results of estimation on each frame (and the accuracy)

    

    std::ofstream ofs;
    if(dataset_type==0){
        std::cout << "file name: accuracy_nr_oxford_"<<append<< std::endl;
        ofs.open("accuracy_nr_oxford_" + append + ".csv", std::ios::out);
    }
    if(dataset_type==1){
        std::cout << "file name: accuracy_nr_mulran_"<<append<< std::endl;
        ofs.open("accuracy_nr_mulran_" + append + ".csv", std::ios::out);
        }
    // ofs << "x,y,yaw,gtx,gty,gtyaw,time1,time2,xmd,ymd,yawmd,xdopp,ydopp,yawdopp\n";
    ofs << "x,y,yaw,gtx,gty,time1,xmd,ymd,yawmd,xdopp,ydopp,yawdopp\n";
    // Create ORB feature detector
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->setPatchSize(patch_size);
    detector->setEdgeThreshold(patch_size);
    // BRUTEFORCE_HAMMING for ORB descriptors FLANNBASED for cen2019 descriptors
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    cv::Mat img1, img2, desc1, desc2;
    std::vector<cv::KeyPoint> kp1, kp2;
    Eigen::MatrixXd targets, cart_targets1, cart_targets2;
    std::vector<int64_t> t1, t2;

    std::vector<int64_t> times;
    std::vector<double> azimuths;
    std::vector<bool> valid;
    cv::Mat fft_data;

    for (uint i = 0; i < radar_files.size() - 1; ++i) {
        std::cout << i << "/" << radar_files.size() << std::endl;
        if (i > 0) {
            t1 = t2; desc1 = desc2.clone(); cart_targets1 = cart_targets2;
            kp1 = kp2; img2.copyTo(img1);
        }
        if(dataset_type == 0)
            load_radar(datadir + "/" + radar_files[i], times, azimuths, valid, fft_data, navtech_version);
        else if(dataset_type == 1)
            load_radar2(datadir + "/" + radar_files[i], times, azimuths, valid, fft_data, navtech_version);

        cv::Mat fft_data_nr = noise_removal(azimuths, fft_data, multipath_thres, radar_resolution, cart_resolution, cart_pixel_width, interp, navtech_version, dataset_type, ang1, ang2, recon_dist);
        // return 0;

        if (keypoint_extraction == 0)
            cen2018features_nr(fft_data,fft_data_nr, zq, sigma_gauss, min_range, targets);
        if (keypoint_extraction == 1)
            cen2019features(fft_data, max_points, min_range, targets);
        if (keypoint_extraction == 0 || keypoint_extraction == 1) {
            radar_polar_to_cartesian(azimuths, fft_data_nr, radar_resolution, cart_resolution, cart_pixel_width, interp, img2, CV_8UC1, navtech_version);  // NOLINT

            // cv::imshow("ori",img2);
            polar_to_cartesian_points(azimuths, times, targets, radar_resolution, cart_targets2, t2);
            convert_to_bev(cart_targets2, cart_resolution, cart_pixel_width, patch_size, kp2, t2);
            // cen2019descriptors(azimuths, cv::Size(fft_data.cols, fft_data.rows), targets, cart_targets2,
                // radar_resolution, 0.3456, 722, desc2, navtech_version);
            detector->compute(img2, kp2, desc2);
        }
        if (keypoint_extraction == 2) {
            detector->detect(img2, kp2);
            detector->compute(img2, kp2, desc2);
            convert_from_bev(kp2, cart_resolution, cart_pixel_width, cart_targets2);
            getTimes(cart_targets2, azimuths, times, t2);
        }
        if (i == 0)
            continue;
        // Match keypoint descriptors
        std::vector<std::vector<cv::DMatch>> knn_matches;
        // matcher->knnMatch(desc1, desc2, knn_matches, 2);

        matcher->radiusMatch(desc1, desc2, knn_matches, 40);

        // Filter matches using nearest neighbor distance ratio (Lowe, Szeliski)
        std::vector<cv::DMatch> good_matches;
        for (uint j = 0; j < knn_matches.size(); ++j) {
            if (!knn_matches[j].size())
                continue;
            if (knn_matches[j][0].distance < nndr * knn_matches[j][1].distance) {
                good_matches.push_back(knn_matches[j][0]);
            }
        }
        // Convert the good key point matches to Eigen matrices
        Eigen::MatrixXd p1 = Eigen::MatrixXd::Zero(2, good_matches.size());
        Eigen::MatrixXd p2 = p1;
        std::vector<int64_t> t1prime = t1, t2prime = t2;
        for (uint j = 0; j < good_matches.size(); ++j) {
            p1(0, j) = cart_targets1(0, good_matches[j].queryIdx);
            p1(1, j) = cart_targets1(1, good_matches[j].queryIdx);
            p2(0, j) = cart_targets2(0, good_matches[j].trainIdx);
            p2(1, j) = cart_targets2(1, good_matches[j].trainIdx);
            t1prime[j] = t1[good_matches[j].queryIdx];
            t2prime[j] = t2[good_matches[j].trainIdx];
        }
        t1prime.resize(good_matches.size());
        t2prime.resize(good_matches.size());

        std::vector<std::string> parts;
        boost::split(parts, radar_files[i], boost::is_any_of("."));
        int64 time1 = std::stoll(parts[0]);
        boost::split(parts, radar_files[i + 1], boost::is_any_of("."));
        int64 time2 = std::stoll(parts[0]);
        double delta_t = (time2 - time1) / 1000000.0;

        // Compute the transformation using RANSAC
        Ransac ransac(p2, p1, ransac_threshold, inlier_ratio, max_iterations);
        srand(i);
        ransac.computeModel();
        Eigen::MatrixXd T;  // T_1_2
        ransac.getTransform(T);

        // Compute the transformation using motion-distorted RANSAC
        MotionDistortedRansac mdransac(p2, p1, t2prime, t1prime, md_threshold, inlier_ratio, max_iterations);
        mdransac.setMaxGNIterations(max_gn_iterations);
        mdransac.correctForDoppler(false);
        srand(i);
        mdransac.computeModel();
        Eigen::MatrixXd Tmd;
        mdransac.getTransform(delta_t, Tmd);
        Tmd = Tmd.inverse();

        // Eigen::VectorXd wbar;
        // mdransac.getMotion(wbar);

        // MDRANSAC + Doppler
        mdransac.correctForDoppler(true);
        mdransac.setDopplerParameter(beta);
        srand(i);
        mdransac.computeModel();
        Eigen::MatrixXd Tmd2 = Eigen::MatrixXd::Zero(4, 4);
        mdransac.getTransform(delta_t, Tmd2);
        Tmd2 = Tmd2.inverse();

        // Retrieve the ground truth to calculate accuracy
        // std::vector<float> gtvec;
        // if (!get_groundtruth_odometry(gt, time1, time2, gtvec)) {
        //     std::cout << "ground truth odometry for " << time1 << " " << time2 << " not found... exiting." << std::endl;
        //     return 0;
        // }
        std::vector<double> gtvec;
        if (!get_groundtruth_odometry3(gt, time1, gtvec)) {
            std::cout << "ground truth odometry for " << time1 << " not found... exiting." << std::endl;
            return 0;
        }
        float yaw = -1 * asin(T(0, 1));
        float yaw2 = -1 * asin(Tmd(0, 1));
        float yaw3 = -1 * asin(Tmd2(0, 1));

        // // Write estimated and ground truth transform to the csv file
        // ofs << T(0, 2) << "," << T(1, 2) << "," << yaw << ","; // x y yaw
        // // ofs << gtvec[0] << "," << gtvec[1] << "," << gtvec[5] << ",";
        // ofs << gtvec[0] << "," << gtvec[1]; // << "," << gtvec[5] << ",";
        // ofs << time1 << "," << time2 << "," << Tmd(0, 3) << "," << Tmd(1, 3) << "," <<  yaw2 << ",";
        // // motion compensation + doppler correction
        // ofs << Tmd2(0, 3) << "," << Tmd2(1, 3) << "," << yaw3 << "\n";

        // Write estimated and ground truth transform to the csv file
        ofs << T(0, 2) << "," << T(1, 2) << "," << yaw << ","; // x y yaw
        // ofs << gtvec[0] << "," << gtvec[1] << "," << gtvec[5] << ",";
        ofs << gtvec[0] << "," << gtvec[1] << ","; // << gtvec[5] << ",";
        ofs << time1 << "," << Tmd(0, 3) << "," << Tmd(1, 3) << "," <<  yaw2 << ",";
        // motion compensation + doppler correction
        ofs << Tmd2(0, 3) << "," << Tmd2(1, 3) << "," << yaw3 << "\n";

    //     cv::Mat img_matches;
    //     cv::drawMatches(img1, kp1, img2, kp2, good_matches, img_matches, cv::Scalar::all(-1),
    //              cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //     cv::imshow("good", img_matches);
    //     cv::waitKey(0);
    }
    

    return 0;
}
