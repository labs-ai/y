#include "cl_wrapper.hpp"
#include "oclUtils.h"

#ifdef WIN32
#pragma warning(disable:4996)
#endif

bool CLSetup::init(const char *gpuDevType, char* deviceName)
{
	if (getPlatformID(gpuDevType)) {
	
		getDeviceID(deviceName);
		getContextnQueue();
		return true;
	}
	else
		return false;
}

bool CLSetup::getPlatformID(const char *gpuDevType)
{

	cl_uint num_of_platforms = 0;
	// get total number of available platforms:
	cl_int err = CL_SUCCESS;
	bool platform_found = false;
	err = clGetPlatformIDs(0, 0, &num_of_platforms);

	if (num_of_platforms == 0) {
		
		printf("ERROR - No OpenCL platforms found !\n");
		return false;
	}
	
	cl_platform_id* platforms = new cl_platform_id[num_of_platforms];
	// get IDs for all platforms:
	err = clGetPlatformIDs(num_of_platforms, platforms, 0);

	for (cl_uint i = 0; i < num_of_platforms; ++i)
	{
		// Get the length for the i-th platform name
		size_t platform_name_length = 0;
		err = clGetPlatformInfo(
			platforms[i],
			CL_PLATFORM_NAME,
			0,
			0,
			&platform_name_length
		);

		// Get the name itself for the i-th platform
		char* platform_name = new char[platform_name_length];
		err = clGetPlatformInfo(
			platforms[i],
			CL_PLATFORM_NAME,
			platform_name_length,
			platform_name,
			0
		);

		// decide if this i-th platform is what we are looking for
		// we select the first one matched skipping the next one if any
		if (err == CL_SUCCESS && strstr(platform_name, gpuDevType)) //"AMD" /*"NVIDIA"*/ /*"Intel(R) OpenCL"*/) &&
			//selected_platform_index == num_of_platforms)
		{
			_platformID = platforms[i];
			platform_found = true;
		}

		delete[] platform_name;
	}

	return platform_found;
}

void CLSetup::getDeviceID(char *devName)
{
    /// !TODO: For Multiple Devices
    _status = clGetDeviceIDs(_platformID,CL_DEVICE_TYPE_GPU, 1, NULL, &_numDevices);
    DEBUG_CL(_status);
    std::cout<<"CL_COMPUTE DEVICES: "<<_numDevices<<std::endl;
    _status = clGetDeviceIDs(_platformID, CL_DEVICE_TYPE_GPU, 1, &_deviceID, NULL);
    DEBUG_CL(_status);
    std::cout<<"CL_DEVICE_ID: "<<_deviceID<<std::endl;

	char device_string[1024];
	clGetDeviceInfo(_deviceID, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
	strcpy(devName, device_string);

    // Getting some information about the device
    // Getting some information about the device

    oclPrintDevInfo(LOGCONSOLE, _deviceID);
    clGetDeviceInfo(_deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &_maxComputeUnits, NULL);
    std::cout<<"CL_DEVICE_MAX_COMPUTE_UNITS: "<<_maxComputeUnits<<std::endl;
    clGetDeviceInfo(_deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &_maxWorkGroupSize, NULL);
    clGetDeviceInfo(_deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &_maxMemAllocSize, NULL);
    clGetDeviceInfo(_deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &_globalMemSize, NULL);
    clGetDeviceInfo(_deviceID, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &_constMemSize, NULL);
    clGetDeviceInfo(_deviceID, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &_localMemSize, NULL);
    ///!TODO:Add get KernelInfo APIs

}

void CLSetup::getContextnQueue()
{
	cl_command_queue_properties queueProps = 0;// CL_QUEUE_PROFILING_ENABLE;
    _context = clCreateContext(NULL, 1, &_deviceID, NULL, NULL, &_status);
    DEBUG_CL(_status);
    _queue = clCreateCommandQueue(_context, _deviceID, queueProps, &_status);
    DEBUG_CL(_status);
}

///
/// \brief CLSetup::createProgram
/// \param kernelFilePath
/// \return
///
Program *CLSetup::createProgram(std::vector<std::string> kernelFilePath)
//Program *CLSetup::createProgram(std::string& kernelFilePath)
{
    ///!TODO: Add support for char** along with string
    Program* tmp = new Program(kernelFilePath, &_context, &_queue,
                               &_deviceID);
    return tmp;
}

OCLBuffer* CLSetup::createBuffer(const size_t size, const cl_mem_flags flags,
                              void *hostMem)
{
    cl_mem buff = clCreateBuffer(_context,flags, size, hostMem ,&_status);
    if(_status == CL_SUCCESS)
    {
        OCLBuffer* ret = new OCLBuffer(buff, &_queue);
        return ret;
    }
    DEBUG_CL(_status);
    if(_status != CL_SUCCESS)
    	printf("createBuffer error : %s", getCLErrorString(_status));
    return NULL; //TODO: Return custom status value
}



















