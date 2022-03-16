/*
 *   pcl_utils.cpp - Some convenience functions for using data from
 *       Joint 2D-3D-Semantic Data for Indoor Scene Understanding
 *       in conjunction with Point Cloud Library (PCL)
 *
 *      Website: 3dsemantics.stanford.edu
 *      Paper: https://arxiv.org/pdf/1702.01105.pdf 
 *  
 *  Usage: Copy or include the code
 */
#ifndef SEMANTIC2D3D_UTILS_H
#define SEMANTIC2D3D_UTILS_H

#include <iostream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <pcl/common/eigen.h>

namespace utils
{

    /** Gets the camera translation from a pose file ptree and returns it**/
    inline Eigen::Affine3d
    getCameraTranslation( const boost::property_tree::ptree &pt ){
        // Read in camera translation
        double tr[3];
        int i = 0;
        boost::property_tree::ptree::const_iterator end = pt.get_child("camera_location").end();
        for (boost::property_tree::ptree::const_iterator it = pt.get_child("camera_location").begin(); it != end; ++it, ++i) {
            tr[i] = it->second.get_value<double>();
        }
        Eigen::Affine3d translation(Eigen::Translation3d(tr[0], tr[1], tr[2]));
        return translation;
    }

    /** Gets the camera rotation from a pose file ptree and returns it**/
    inline Eigen::Affine3d
    getCameraRotation( const boost::property_tree::ptree &pt ){
        // Read in the camera euler angles
        int i = 0;
        double euler_angles[3];
        boost::property_tree::ptree::const_iterator end = pt.get_child("camera_rotation_final").end();
        for (boost::property_tree::ptree::const_iterator it = pt.get_child("camera_rotation_final").begin(); it != end; ++it, ++i) {
            euler_angles[i] = it->second.get_value<double>();
        }
        //Roll pitch and yaw in Radians
        double roll = euler_angles[2], pitch = euler_angles[1], yaw = euler_angles[0];
        Eigen::Quaterniond q;
        q = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitZ())
            * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
            * Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitX());
        Eigen::Affine3d rotation( q );
        return rotation;
    }

    /** Load in the json pose files into a boost::ptree **/
    inline boost::property_tree::ptree 
    loadPoseFile( const std::string &json_filename ) {
        // Read in view_dict
        boost::property_tree::ptree pt;
        std::ifstream in(json_filename.c_str());
        std::stringstream buffer;
        buffer << in.rdbuf();    
        read_json(buffer, pt);
        return pt;
    }
    
    /** Debugging function to print out a boost::ptree **/
    void print(boost::property_tree::ptree const& pt){
    using boost::property_tree::ptree;
    ptree::const_iterator end = pt.end();
    for (ptree::const_iterator it = pt.begin(); it != end; ++it) {
        std::cout << it->first << ": " << it->second.get_value<std::string>() << std::endl;
        print(it->second);
    }
}
} // END namespace

#endif  // #ifndef SEMANTIC2D3D_UTILS_H
