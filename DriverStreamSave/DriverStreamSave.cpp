// ConsoleApplication1.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <direct.h>
#include <librealsense/rs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace rs;


// Window size and frame rate
int const INPUT_WIDTH = 320;
int const INPUT_HEIGHT = 240;
int const FRAMERATE = 30;

int const SAVE_TIME = 10;

// Named windows
char* const WINDOW_DEPTH = "Depth Image";
char* const WINDOW_RGB = "RGB Image";
char* const WINDOW_IR = "IR Image";


context 	_rs_ctx;
device* 	_rs_camera = NULL;
intrinsics 	_depth_intrin;
intrinsics  _color_intrin;
bool 		_loop = true;
bool 		_saving = false;
int			_loop_count = 0;

cv::Mat matColor[FRAMERATE*SAVE_TIME+1];
cv::Mat matDepth[FRAMERATE*SAVE_TIME + 1];
cv::Mat matIR1[FRAMERATE*SAVE_TIME + 1];
cv::Mat matIR2[FRAMERATE*SAVE_TIME + 1];

// Initialize the application state. Upon success will return the static app_state vars address
bool initialize_streaming()
{
	bool success = false;
	if (_rs_ctx.get_device_count() > 0)
	{
		_rs_camera = _rs_ctx.get_device(0);

		// set before stream
		//_rs_camera->set_option(rs::option::r200_depth_clamp_max, 1.0f); //Minimum depth in current depth units that will be output. Any values less than "Max Depth" will be mapped to 0 during the conversion between disparity and depth. Set before streaming. 
		//_rs_camera->set_option(rs::option::r200_disparity_multiplier, 1.0f); //Disparity scale factor used when in disparity output mode. Can only be set before streaming.
		//_rs_camera->set_option(rs::option::r200_depth_units, 1.0f); //Micrometers per increment in integer depth values. 1000 is default (mm scale). Set before streaming.
		//_rs_camera->set_option(rs::option::r200_depth_clamp_min, 1.0f); //Minimum depth in current depth units that will be output. Any values less than "Min Depth" will be mapped to 0 during the conversion between disparity and depth. Set before streaming.

		_rs_camera->enable_stream(rs::stream::color, INPUT_WIDTH, INPUT_HEIGHT, rs::format::rgb8, FRAMERATE);
		_rs_camera->enable_stream(rs::stream::depth, INPUT_WIDTH, INPUT_HEIGHT, rs::format::z16, FRAMERATE);
		_rs_camera->enable_stream(rs::stream::infrared, INPUT_WIDTH, INPUT_HEIGHT, rs::format::y8, FRAMERATE);
		_rs_camera->enable_stream(rs::stream::infrared2, INPUT_WIDTH, INPUT_HEIGHT, rs::format::y8, FRAMERATE);

		_rs_camera->start();

		_rs_camera->set_option(rs::option::r200_emitter_enabled, 1.0f); //Enables/disables R200 emitter 
		_rs_camera->set_option(rs::option::r200_lr_auto_exposure_enabled, 1.0f); //Enable/disable R200 auto-exposure. This will affect both IR and depth images. 

		//_rs_camera->set_option(rs::option::r200_lr_gain, 1.0f); //IR image gain 
		//_rs_camera->set_option(rs::option::r200_lr_exposure, 1.0f); //This control allows manual adjustment of the exposure time value for the L/R imagers. 
		//_rs_camera->set_option(rs::option::r200_disparity_shift, 1.0f); //{0 - 512}. Can only be set before streaming starts.
		//_rs_camera->set_option(rs::option::r200_auto_exposure_mean_intensity_set_point, 1.0f); //Mean intensity set point. Requires the r200_lr_auto_exposure_enabled option to be set to 1
		//_rs_camera->set_option(rs::option::r200_auto_exposure_bright_ratio_set_point, 1.0f); //Bright ratio set point. Requires the r200_lr_auto_exposure_enabled option to be set to 1.
		//_rs_camera->set_option(rs::option::r200_auto_exposure_kp_gain, 1.0f); //Kp gain. Requires the r200_lr_auto_exposure_enabled option to be set to 1
		//_rs_camera->set_option(rs::option::r200_auto_exposure_kp_exposure, 1.0f); //Kp exposure. Requires the r200_lr_auto_exposure_enabled option to be set to 1
		//_rs_camera->set_option(rs::option::r200_auto_exposure_kp_dark_threshold, 1.0f); //Kp dark threshold. Requires the r200_lr_auto_exposure_enabled option to be set to 1. 
		//_rs_camera->set_option(rs::option::r200_auto_exposure_top_edge, 1.0f); //Auto-exposure region-of-interest top edge (in pixels). Requires the r200_lr_auto_exposure_enabled option to be set to 1
		//_rs_camera->set_option(rs::option::r200_auto_exposure_bottom_edge, 1.0f); //Auto-exposure region-of-interest bottom edge (in pixels). Requires the r200_lr_auto_exposure_enabled option to be set to 1
		//_rs_camera->set_option(rs::option::r200_auto_exposure_left_edge, 1.0f); //Auto-exposure region-of-interest left edge (in pixels). Requires the r200_lr_auto_exposure_enabled option to be set to 1.
		//_rs_camera->set_option(rs::option::r200_auto_exposure_right_edge, 1.0f); //Auto-exposure region-of-interest right edge (in pixels). Requires the r200_lr_auto_exposure_enabled option to be set to 1.
		//_rs_camera->set_option(rs::option::r200_depth_control_estimate_median_decrement, 1.0f); //Value to subtract when estimating the median of the correlation surface
		//_rs_camera->set_option(rs::option::r200_depth_control_estimate_median_increment, 1.0f); //Value to add when estimating the median of the correlation surface
		//_rs_camera->set_option(rs::option::r200_depth_control_median_threshold, 1.0f);	//Threshold: by how much the winning score exceeds the median
		//_rs_camera->set_option(rs::option::r200_depth_control_score_minimum_threshold, 1.0f); //Minimum correlation score that is considered acceptable
		//_rs_camera->set_option(rs::option::r200_depth_control_score_maximum_threshold, 1.0f); // Maximum correlation score that is considered acceptable 
		//_rs_camera->set_option(rs::option::r200_depth_control_texture_count_threshold, 1.0f); //Parameter for determining whether the texture in the region is sufficient to justify a depth result 
		//_rs_camera->set_option(rs::option::r200_depth_control_texture_difference_threshold, 1.0f); // Parameter for determining whether the texture in the region is sufficient to justify a depth result
		//_rs_camera->set_option(rs::option::r200_depth_control_second_peak_threshold, 1.0f); //Threshold: how much the minimum correlation score must differ from the next best score.
		//_rs_camera->set_option(rs::option::r200_depth_control_neighbor_threshold, 1.0f); //Neighbor threshold value for depth calculation 
		//_rs_camera->set_option(rs::option::r200_depth_control_lr_threshold, 1.0f);	//Left-right threshold value for depth calculation 

		success = true;
	}
	return success;
}



/////////////////////////////////////////////////////////////////////////////
// If the left mouse button was clicked on either image, stop streaming and close windows.
/////////////////////////////////////////////////////////////////////////////
static void onMouse(int event, int x, int y, int, void* window_name)
{
	if (event == cv::EVENT_RBUTTONDOWN)
	{
		_loop = false;
	}
	else if (event == cv::EVENT_LBUTTONDOWN) 
	{
		_saving = true;
	}
	else if (event == cv::EVENT_MBUTTONDOWN)
	{
		_rs_camera->set_option(rs::option::r200_lr_auto_exposure_enabled, 1.0f);
	}
}



/////////////////////////////////////////////////////////////////////////////
// Create the depth and RGB windows, set their mouse callbacks.
// Required if we want to create a window and have the ability to use it in
// different functions
/////////////////////////////////////////////////////////////////////////////
void setup_windows()
{
	cv::namedWindow(WINDOW_DEPTH, 0);
	cv::namedWindow(WINDOW_RGB, 0);
	cv::namedWindow(WINDOW_IR, 0);

	cv::setMouseCallback(WINDOW_DEPTH, onMouse, WINDOW_DEPTH);
	cv::setMouseCallback(WINDOW_RGB, onMouse, WINDOW_RGB);
	cv::setMouseCallback(WINDOW_IR, onMouse, WINDOW_IR);
}



/////////////////////////////////////////////////////////////////////////////
// Called every frame gets the data from streams and displays them using OpenCV.
/////////////////////////////////////////////////////////////////////////////
bool display_next_frame()
{
	auto start = std::chrono::steady_clock::now();

	// Get current frames intrinsic data.
	_depth_intrin = _rs_camera->get_stream_intrinsics(rs::stream::depth);
	_color_intrin = _rs_camera->get_stream_intrinsics(rs::stream::color);

	// Create depth image
	cv::Mat depth16(_depth_intrin.height,
		_depth_intrin.width,
		CV_16U,
		(uchar *)_rs_camera->get_frame_data(rs::stream::depth));

	// Create color image
	cv::Mat rgb(_color_intrin.height,
		_color_intrin.width,
		CV_8UC3,
		(uchar *)_rs_camera->get_frame_data(rs::stream::color));

	// Create color image
	cv::Mat ir(_depth_intrin.height,
		_depth_intrin.width,
		CV_8UC1,
		(uchar *)_rs_camera->get_frame_data(rs::stream::infrared));
	// Create color image
	cv::Mat ir2(_depth_intrin.height,
		_depth_intrin.width,
		CV_8UC1,
		(uchar *)_rs_camera->get_frame_data(rs::stream::infrared2));

	// < 800
	cv::Mat depth8u = depth16;
	depth8u.convertTo(depth8u, CV_8UC1, 255.0 / 1000);

	imshow(WINDOW_DEPTH, depth8u);
	cvWaitKey(1);

	imshow(WINDOW_IR, ir);
	cvWaitKey(1);

	cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
	imshow(WINDOW_RGB, rgb);
	cvWaitKey(1);

	if (_saving)
	{
		cout << "saving stream : " << _loop_count++ << " : ";
		matColor[_loop_count] = rgb.clone();
		matDepth[_loop_count] = depth16.clone();
		matIR1[_loop_count] = ir.clone();
		matIR2[_loop_count] = ir2.clone();
	}

	auto finish = std::chrono::steady_clock::now();
	double elapsed_seconds = std::chrono::duration_cast<
		std::chrono::duration<double> >(finish - start).count();
	cout << elapsed_seconds << endl;

	return true;
}

void init_variables()
{
	_loop_count = 0;
	_loop = true;
	_saving = false;
}

void save_frames_to_disk(string user, int point)
{
	auto start = std::chrono::steady_clock::now();
	cout << "Saving frames to disk! ";
	string filename = user;
	//filename += "\\";
	//filename += user;
	filename += "_";
	filename += std::to_string(point);
	filename += "_";

	string savename;
	for (int i = 0; i < FRAMERATE * SAVE_TIME; i++)
	{
		savename = filename;
		savename += "rgb";
		savename += "_";
		savename += std::to_string(i);
		savename += ".jpeg";
		imwrite(savename, matColor[i]);

		savename = filename;
		savename += "depth";
		savename += "_";
		savename += std::to_string(i);
		savename += ".yaml";
		cv::FileStorage file(savename, cv::FileStorage::WRITE);
		file << "depth" << matDepth[i];

		savename = filename;
		savename += "depth";
		savename += "_";
		savename += std::to_string(i);
		savename += ".jpeg";
		imwrite(savename, matDepth[i]);

		savename = filename;
		savename += "ir1";
		savename += "_";
		savename += std::to_string(i);
		savename += ".jpeg";
		imwrite(savename, matIR1[i]);

		savename = filename;
		savename += "ir2";
		savename += "_";
		savename += std::to_string(i);
		savename += ".jpeg";
		imwrite(savename, matIR2[i]);
	}
	auto finish = std::chrono::steady_clock::now();
	double elapsed_seconds = std::chrono::duration_cast<
		std::chrono::duration<double> >(finish - start).count();
	cout << elapsed_seconds << endl;
}

/////////////////////////////////////////////////////////////////////////////
// Main function
/////////////////////////////////////////////////////////////////////////////
int main() try
{
	char key = 0;
	string user;
	rs::log_to_console(rs::log_severity::warn);

	if (!initialize_streaming())
	{
		std::cout << "Unable to locate a camera" << std::endl;
		rs::log_to_console(rs::log_severity::fatal);
		return EXIT_FAILURE;
	}

	setup_windows();

	do {
		cout << "enter name : ";
		std::getline(std::cin, user);

		if (!user.empty())
		{
			//mkdir(user.c_str());

			while (_loop) 
			{
				int point = 0;
				cout << "Point to save: ";
				cin >> point;

				if (point == 0)
				{
					break;
				}

				init_variables();

				// Loop until someone left clicks on either of the images in either window.
				while (_loop && _loop_count < 300)
				{
					if (_rs_camera->is_streaming())
						_rs_camera->wait_for_frames();

					display_next_frame();
				}

				save_frames_to_disk(user, point);
			}
		}
	} while (!user.empty());

	_rs_camera->stop();

	cv::destroyAllWindows();

	return EXIT_SUCCESS;
}
catch (const rs::error & e)
{
	std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
	return EXIT_FAILURE;
}
catch (const std::exception & e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}
