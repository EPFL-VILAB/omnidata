/* Desc: narf_interst_image.cpp
  Author: Sasha Sax, CVGL
  Desc: Reads in a depth image and saves out an interest image. An interest image contains
    the 'soft' output of a keypoint detector, in this case NARF 
    (paper: https://pdfs.semanticscholar.org/e070/1662a370622a1cdbb7c6a83bbede3d0e6c23.pdf)
    Specifically, the interest image contains intensity values for each pixel before a 
    hard yes/no decision is made as to whether that pixel is a keypoint. This will probably
    provide richer signals to a learning model and that is why we use the interest image 
    instead of just keypoints. 

  Usage: ./narf_interest_image.bin [options] <inputDepthImage.png> <outputInterestImage.png>

    Options:
    -------------------------------------------
    -r <int>     Image resolution in pixels
    -f <float>   Focal length in pixels
    -d <float>   The distance that corresponds to a 1-pixel-intensity difference - default 0.00195312)
    -m           Treat all unseen points to max range
    -s <float>   support size for the interest points (diameter of the used sphere - default 0.2)
    -o <0/1>     switch rotational invariant version of the feature on/off (default 1)
    -v <0/1>     Be verbose and visualize the output -  (default 0)
    -h           this help

  Requires (to be run):
    - generate_points.py
    - create_depth_images.py
*/
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/features/narf_descriptor.h>
// #include <pcl/keypoints/narf_keypoint.h>
#include "pcl_narf.hpp"
#include <pcl/range_image/range_image_planar.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

// This needs to be included before GIL because cruft
#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL
#include <boost/gil/gil_all.hpp>
#include <boost/gil/gray.hpp>
#include <boost/gil/extension/io/png_dynamic_io.hpp>
#include <iostream>
using namespace boost::gil;

// #include <stdint.h>

using boost::property_tree::ptree;

float support_size = 0.2f; // Use points w/in this many meters to detect keypoints
bool set_unseen_to_max_range = false;
bool rotation_invariant = true;
float sensitivity = 1.0f/512.0f;
int resolution = 256;
float focal_length = 242.9311106566421;
bool visualize = false;

void 
printUsage (const char* progName)
{
  std::cout << "\n\nUsage: "<<progName<<" [options] <inputDepthImage.png> <outputInterestImage.png>\n\n"
            << "Options:\n"
            << "-------------------------------------------\n"
            << "-r <int>     Image resolution in pixels\n"
            << "-f <float>   Focal length in pixels\n"
            << "-d <float>   The distance that corresponds to a 1-pixel-intensity difference - "
                                                                  "default "<<sensitivity<<")\n"
            << "-m           Treat all unseen points to max range\n"
            << "-s <float>   support size for the interest points (diameter of the used sphere - "
                                                                  "default "<<support_size<<")\n"
            << "-o <0/1>     switch rotational invariant version of the feature on/off"
            <<               " (default "<< (int)rotation_invariant<<")\n"
            << "-v <0/1>     Be verbose and visualize the output - "
            <<               " (default "<< (int)visualize<<")\n"
            << "-h           this help\n"
            << "\n\n";
}


// Uncomment this if building and using visualizations
void 
setViewerPose (pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f (0, 0, 0);
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f (0, 0, -1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f (0, 1, 0);
  viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}

struct PixelInserter{
        std::vector<uint16_t>* storage;
        PixelInserter(std::vector<uint16_t>* s) : storage(s) {}
        void operator()(boost::gil::gray16_pixel_t p) const {
                storage->push_back(boost::gil::at_c<0>(p));
        }
};


int
main(int argc, char** argv)
{   

    // --------------------------------------
    // -----Parse Command Line Arguments-----
    // --------------------------------------
    if (pcl::console::parse (argc, argv, "-v", visualize) >= 0)
        if(visualize)
            std::cout << "Setting visualization to "<<visualize<<".\n";
    if (pcl::console::find_argument (argc, argv, "-h") >= 0)
    {
        printUsage (argv[0]);
        return 0;
    }
    if (pcl::console::find_argument (argc, argv, "-m") >= 0 && visualize)
    {
        set_unseen_to_max_range = true;
        std::cout << "Setting unseen values in range image to maximum range readings.\n";
    }
    if (pcl::console::parse (argc, argv, "-o", rotation_invariant) >= 0 && visualize)
        std::cout << "Switching rotation invariant feature version "<< (rotation_invariant ? "on" : "off")<<".\n";
    int tmp_coordinate_frame;
    if (pcl::console::parse (argc, argv, "-s", support_size) >= 0 && visualize)
        std::cout << "Setting support size to "<<support_size<<".\n";
    if (pcl::console::parse (argc, argv, "-r", resolution) >= 0 && visualize)
        std::cout << "Setting resolution to "<<resolution<<".\n";
    if (pcl::console::parse (argc, argv, "-f", focal_length) >= 0 && visualize)
        std::cout << "Setting focal length to "<<focal_length<<" pixels.\n";
    if (pcl::console::parse (argc, argv, "-d", sensitivity) >= 0 && visualize)
        std::cout << "Setting distance per pixel to "<<sensitivity<<".\n";

    std::vector<int> png_filename_indices = pcl::console::parse_file_extension_argument (argc, argv, "png");
    std::string input_filename = argv[png_filename_indices[0]];
    std::string output_filename = argv[png_filename_indices[1]];
    if(visualize) {
        std::cout << "Input " << input_filename << std::endl;
        std::cout << "Output " << output_filename << std::endl;
        
    }
    double program_start_time = pcl::getTime();
	// Parameters needed by the planar range image object:

	// Image size. Both Kinect and Xtion work at 640x480.
	int imageSizeX = resolution;
	int imageSizeY = resolution;
	// Center of projection. here, we choose the middle of the image.
	float centerX = ((float)resolution) / 2.0f;
	float centerY = ((float)resolution) / 2.0f;
	float focalLengthX = focal_length, focalLengthY = focalLengthX;
    

	// Sensor pose. We just set this to empty.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	Eigen::Affine3f sensorPose = Eigen::Affine3f(Eigen::Translation3f(cloud->sensor_origin_[0],
								 cloud->sensor_origin_[1],
								 cloud->sensor_origin_[2])) *
								 Eigen::Affine3f(cloud->sensor_orientation_);
	
    // Noise level. If greater than 0, values of neighboring points will be averaged.
	// This would set the search radius (e.g., 0.03 == 3cm).
	float noiseLevel = 0.0f;
	float minimumRange = 0.0f; // Minimum range. If set, any point closer to the sensor than this will be ignored.

    

    // ------------------------
    // -----Read range png-----
    // ------------------------  
    double load_start_time = pcl::getTime();
    boost::gil::gray16_image_t image;
    std::vector<uint16_t> storage;
    storage.reserve(resolution * resolution);
    std::vector<float> depthImage;
    depthImage.reserve(resolution * resolution);

    boost::gil::png_read_and_convert_image(input_filename, image);

    // Save image as array and multiply by the pixel sensitivity
    for_each_pixel(const_view(image), PixelInserter(&storage));
    for ( int i = 0; i < storage.size (); i++ )
        depthImage.push_back( (float) storage[i] * sensitivity );

    double load_time = pcl::getTime()-load_start_time;
    if(visualize) std::cout << "Loading took "<<1000.0*load_time<<"ms.\n";


    // -------------------------
    // -----Set Range Image-----
    // -------------------------  
	// Planar range image object.
    boost::shared_ptr<pcl::RangeImagePlanar> range_image_planar_ptr (new pcl::RangeImagePlanar);
    pcl::RangeImagePlanar& range_image_planar = *range_image_planar_ptr;
	range_image_planar.setDepthImage(&depthImage[0], imageSizeX, imageSizeY,
			centerX, centerY, focalLengthX, focalLengthX );


    // --------------------------------
    // -----Extract NARF keypoints-----
    // --------------------------------
    if (set_unseen_to_max_range)
      range_image_planar.setUnseenToMaxRange ();
    
    pcl::RangeImageBorderExtractor range_image_border_extractor;
    pcl::NarfKeypoint2 narf_keypoint_detector;
    narf_keypoint_detector.setRangeImageBorderExtractor (&range_image_border_extractor);
    narf_keypoint_detector.setRangeImage (&range_image_planar);
    narf_keypoint_detector.getParameters ().support_size = support_size;
    narf_keypoint_detector.getParameters ().calculate_sparse_interest_image = false;
    narf_keypoint_detector.getParameters ().use_recursive_scale_reduction = true;
    // narf_keypoint_detector.getParameters ().do_non_maximum_suppression = false;

    pcl::PointCloud<int> keypoint_indices;
    double keypoint_extraction_start_time = pcl::getTime();
    // narf_keypoint_detector.compute (keypoint_indices);
    // narf_keypoint_detector.calculateInterestImage ();
    float* interest_ptr = narf_keypoint_detector.getInterestImage ();
    double keypoint_extraction_time = pcl::getTime()-keypoint_extraction_start_time;
    if(visualize) 
        std::cout << "Found "<<keypoint_indices.points.size ()<<" key points. "
              << "This took "<<1000.0*keypoint_extraction_time<<"ms.\n";


    double program_time = pcl::getTime()-program_start_time;
    if(visualize) 
        std::cout <<  "Program took "<<1000.0*program_time<<"ms.\n";
    

    // -----------------------------
    // -----Save interest image-----
    // -----------------------------
    // One option is to write out the interest points as a json
    // ptree pt;
    // ptree children;
    // for( int i=0; i < keypoint_indices.points.size(); i++ ) {
    //     ptree interest_node;
    //     interest_node.put("", keypoint_indices.points[i]);
    //     children.push_back( std::make_pair("", interest_node) );
    // }
    // pt.add_child("Interests", children);
    // write_json(output_filename + "kps.json", pt);
    
    // But better to save the interest points as a png
    const gray16_view_t & mViewer = view(image);
    int count_zero = 0;
    int count_nonzero = 0;
    for (int y = 0; y < mViewer.height(); ++y)
    {
        gray16_view_t::x_iterator trIt = mViewer.row_begin(y);
        for (int x = 0; x < mViewer.width(); ++x, ++interest_ptr){
            at_c<0>(trIt[x]) = (*(interest_ptr) * std::pow(2,16)) ;
        }
    }
    boost::gil::png_write_view(output_filename, mViewer);

    if(!visualize)
        return 0;

    // ----------------------------------------------
    // -----Show keypoints in range image widget-----
    // ----------------------------------------------
  	pcl::visualization::RangeImageVisualizer range_image_widget("Planar range image");
	range_image_widget.showRangeImage(range_image_planar);
    for (size_t i=0; i<keypoint_indices.points.size (); ++i){
        range_image_widget.markPoint (keypoint_indices.points[i]%range_image_planar.width,
                                    resolution - 1 - keypoint_indices.points[i]/range_image_planar.width,
                                    pcl::visualization::Vector3ub (0,255,0));
    }


    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    pcl::visualization::PCLVisualizer viewer ("3D Viewer");
    viewer.setBackgroundColor (1, 1, 1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> range_image_color_handler (range_image_planar_ptr, 0, 0, 0);
    viewer.addPointCloud ( range_image_planar_ptr, range_image_color_handler, "range image");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "range image");
    viewer.initCameraParameters ();
    setViewerPose (viewer, range_image_planar.getTransformationToWorldSystem ());
  

    // -------------------------------------
    // -----Show keypoints in 3D viewer-----
    // -------------------------------------
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>& keypoints = *keypoints_ptr;
    keypoints.points.resize (keypoint_indices.points.size ());
    for (size_t i=0; i<keypoint_indices.points.size (); ++i)
        keypoints.points[i].getVector3fMap () = range_image_planar.points[keypoint_indices.points[i]].getVector3fMap ();
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler (keypoints_ptr, 0, 255, 0);
    viewer.addPointCloud<pcl::PointXYZ> (keypoints_ptr, keypoints_color_handler, "keypoints");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints");
  
	while (!viewer.wasStopped())
	{
		range_image_widget.spinOnce();
		// Sleep 100ms to go easy on the CPU.
		pcl_sleep(0.1);
	}

}


