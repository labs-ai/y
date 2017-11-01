
/*Copyright 2017 Sateesh Pedagadi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.*/

#include "YoloOCLDNN.h"

#ifdef __linux__
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#endif

//#define strcpy strcpy_s

float BBOX_COLORS[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

LOGGERCALLBACK externLogFunc;

void WaitMilliSecs(int mSecs) {

#ifdef WIN32
	::Sleep(mSecs);
#elif __linux__
	usleep(mSecs * 1000);
#endif

}

#ifdef WIN32

#include <windows.h>
#include <tchar.h>
#include <stdio.h>


void EnumerateFilesInDirectory(string srcFolder,  vector<string> &fileNames, vector<string> &imageNames) {

	WIN32_FIND_DATA fd;
	string search_path = "";

	for (int i = 0; i < 2; i++) {
	
		if(i == 0)
			search_path = srcFolder + "\\*.jpg";
		else if(i == 1)
			search_path = srcFolder + "\\*.png";


		HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
		if (hFind != INVALID_HANDLE_VALUE) {
			do {
				if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
					imageNames.push_back(fd.cFileName);
					fileNames.push_back(srcFolder + "\\" + fd.cFileName);
				}
			} while (::FindNextFile(hFind, &fd));
			::FindClose(hFind);
		}
	}
}

#elif __linux__

bool IsFileAnImage(const std::string& FileName) {

    printf("IsFileAnImage()\n");
    std::string extension = "";
    if(FileName.find_last_of(".") != std::string::npos)
        extension = FileName.substr(FileName.find_last_of(".")+1);
	
    printf("IsFileAnImage() ext %s\n", extension.c_str());
    if(extension == "jpg" || extension == "png")
	return true;
    else
	return false;
}

void EnumerateFilesInDirectory(string srcFolder,  vector<string> &fileNames, vector<string> &imageNames) {

    struct dirent *entry;
    char filePath[FILENAME_MAX];
    DIR *dir = opendir(srcFolder.c_str());
    if (dir == NULL) {
        return;
    }

    while ((entry = readdir(dir)) != NULL) {
	
	if(IsFileAnImage(std::string(entry->d_name))) {

		sprintf(filePath, "%s/%s", srcFolder.c_str(), entry->d_name);
		fileNames.push_back(std::string(filePath));
		imageNames.push_back(entry->d_name);
	}
    }

    closedir(dir);
}

#endif

float rand_uniform(float minVal, float maxVal)
{
	if (maxVal < minVal) {

		float swap = minVal;
		minVal = maxVal;
		maxVal = swap;
	}
	return ((float)rand() / RAND_MAX * (maxVal - minVal)) + minVal;
}

void ParseDelimitedStrToIntVec(std::string instr, std::vector<int> &outVals) {

	std::stringstream ss(instr);

	int i;

	while (ss >> i){

		outVals.push_back(i);
		if (ss.peek() == ',')
			ss.ignore();
	}
}

void ParseDelimitedStrToFloatVec(std::string instr, std::vector<float> &outVals) {

	std::stringstream ss(instr);

	float i;

	while (ss >> i) {

		outVals.push_back(i);
		if (ss.peek() == ',')
			ss.ignore();
	}
}

EnumYOLODeepNNLayerType MapNNLayerTypeStr(std::string inStr) {

	if (inStr.find("region") != std::string::npos)
		return EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_REGION;
	else if (inStr.find("convolutional") != std::string::npos)
		return EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_CONVOLUTIONAL;
	else if (inStr.find("maxpool") != std::string::npos)
		return EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_MAXPOOL;
	
	return EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_CONVOLUTIONAL;

}

EnumYOLODeepNNActivationType MapNNLayerActivationStr(char *activationStr) {

	if (strcmp(activationStr, "logistic") == 0) 
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_LOGISTIC;
	if (strcmp(activationStr, "loggy") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_LOGGY;
	if (strcmp(activationStr, "relu") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_RELU;
	if (strcmp(activationStr, "elu") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_ELU;
	if (strcmp(activationStr, "relie") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_RELIE;
	if (strcmp(activationStr, "plse") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_PLSE;
	if (strcmp(activationStr, "hardtan") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_HARDTAN;
	if (strcmp(activationStr, "lhtan") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_LHTAN;
	if (strcmp(activationStr, "linear") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_LINEAR;
	if (strcmp(activationStr, "ramp") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_RAMP;
	if (strcmp(activationStr, "leaky") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_LEAKY;
	if (strcmp(activationStr, "tanh") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_TANH;
	if (strcmp(activationStr, "stair") == 0)
		return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_STAIR;

	printf("ERROR : Couldn't find activation function %s, going with ReLU\n", activationStr);

	return EnumYOLODeepNNActivationType::YOLO_DNN_ACTIVATION_TYPE_RELU;
}


YOLONeuralNet::YOLONeuralNet(LOGGERCALLBACK loggerCallback, char *gpuDevType, char* classLabelsFile, char *networkConfigFile,  char *weightsFile,
	bool display, EnumSaveOPType saveOPType, float threshold, float nmsOverlap) {


	logWriteFunc = loggerCallback;
	externLogFunc = loggerCallback;
	strcpy(m_GPUDevType, gpuDevType);
	strcpy(m_ClassLabelsFile, classLabelsFile);
	strcpy(m_NetworkConfigFile, networkConfigFile);
	strcpy(m_WeightsFile, weightsFile);
	m_CairoSurface = NULL;
	m_EnableDisplay = display;
	m_SaveOPType = saveOPType;
	m_DetThreshold = threshold;
	m_NMSOverlap = nmsOverlap;
	m_VideoFileEOS = false;
	m_SinkActive = true;
	m_AVIHeaderWritten = false;
	av_register_all();
	avcodec_register_all();
	avformat_network_init();
}

YOLONeuralNet::~YOLONeuralNet() {

}

bool YOLONeuralNet::ParseNetworkConfiguration() {


	CSimpleIniA::TNamesDepend sections;
	m_IniReader->GetAllSections(sections);
	sections.sort(CSimpleIniA::Entry::LoadOrder());

	if(sections.size() == 0) {

		//LOG error
		return false;
	}

	CSimpleIniA::TNamesDepend::const_iterator i;
	for (i = sections.begin(); i != sections.end(); ++i)
		m_LayerNames.push_back(i->pItem);

		
	memset(m_YOLODeepNN, 0, sizeof(StructYOLODeepNN));
	m_YOLODeepNN->m_TotalLayers = (int)sections.size() - 1;
	m_YOLODeepNN->m_Layers = (StructYOLODeepNNLayer*)calloc(m_YOLODeepNN->m_TotalLayers, sizeof(StructYOLODeepNNLayer));

	m_YOLODeepNN->m_GpuIndex = 0; // TODO : Pass this as part of configuration
	m_YOLODeepNN->m_BatchSize = (int)m_IniReader->GetDoubleValue("net", "batch", 1);
	//int subDivs = (int)m_IniReader->GetDoubleValue("net", "subdivisions", 1);
	m_YOLODeepNN->m_TimeSteps = (int)m_IniReader->GetDoubleValue("net", "time_steps", 1);
	m_YOLODeepNN->m_H = (int)m_IniReader->GetDoubleValue("net", "height", 0);
	m_YOLODeepNN->m_W = (int)m_IniReader->GetDoubleValue("net", "width", 0);
	m_YOLODeepNN->m_C = (int)m_IniReader->GetDoubleValue("net", "channels", 0);
	m_YOLODeepNN->m_Inputs = (int)m_IniReader->GetDoubleValue("net", "inputs", m_YOLODeepNN->m_H * m_YOLODeepNN->m_W * m_YOLODeepNN->m_C);
	
	if (!m_YOLODeepNN->m_Inputs && !(m_YOLODeepNN->m_H && m_YOLODeepNN->m_W && m_YOLODeepNN->m_C)) {

		//LOG Error
		return false;
	}


	return true;
}

bool YOLONeuralNet::PrepareConvolutionalTypeLayer(int sectionIdx, int layerIdx, StructLayerFeedParams *layerFeedParams) {

	int pad = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "pad", 0);
	int padding = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "padding", 0);
	if (pad)
		padding = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "size", 1) / 2;


	char activation_s[512];
	strcpy(activation_s, m_IniReader->GetValue((char*)m_LayerNames[sectionIdx].c_str(), "activation", "logistic"));//, activation_s);

	EnumYOLODeepNNActivationType activation = MapNNLayerActivationStr(activation_s);

	m_YOLODeepNN->m_Layers[layerIdx].m_Flipped = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "flipped", 0);

	m_YOLODeepNN->m_Layers[layerIdx].m_LayerType = EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_CONVOLUTIONAL;

	m_YOLODeepNN->m_Layers[layerIdx].m_H = layerFeedParams->m_H;
	m_YOLODeepNN->m_Layers[layerIdx].m_W = layerFeedParams->m_W;
	m_YOLODeepNN->m_Layers[layerIdx].m_C = layerFeedParams->m_C;
	m_YOLODeepNN->m_Layers[layerIdx].m_N = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "filters", 1);
	m_YOLODeepNN->m_Layers[layerIdx].m_Batch = layerFeedParams->m_Batch;
	m_YOLODeepNN->m_Layers[layerIdx].m_Stride = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "stride", 1); ;
	m_YOLODeepNN->m_Layers[layerIdx].m_Size = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "size", 1); ;
	m_YOLODeepNN->m_Layers[layerIdx].m_Pad = padding;
	m_YOLODeepNN->m_Layers[layerIdx].m_BatchNormalize = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "batch_normalize", 0);

	int weightsLength = m_YOLODeepNN->m_Layers[layerIdx].m_C *
		m_YOLODeepNN->m_Layers[layerIdx].m_N * m_YOLODeepNN->m_Layers[layerIdx].m_Size *
		m_YOLODeepNN->m_Layers[layerIdx].m_Size;

	m_YOLODeepNN->m_Layers[layerIdx].m_Weights = (float*)calloc(weightsLength, sizeof(float));

	m_YOLODeepNN->m_Layers[layerIdx].m_Biases = (float*)calloc(m_YOLODeepNN->m_Layers[layerIdx].m_N, sizeof(float));

	float scale = (float)sqrt(2. / (m_YOLODeepNN->m_Layers[layerIdx].m_Size * m_YOLODeepNN->m_Layers[layerIdx].m_Size * m_YOLODeepNN->m_Layers[layerIdx].m_C));

	for (int i = 0; i < weightsLength; ++i)
		m_YOLODeepNN->m_Layers[layerIdx].m_Weights[i] = scale * rand_uniform(-1, 1);

	m_YOLODeepNN->m_Layers[layerIdx].m_OutH = (m_YOLODeepNN->m_Layers[layerIdx].m_H + 2 * m_YOLODeepNN->m_Layers[layerIdx].m_Pad - m_YOLODeepNN->m_Layers[layerIdx].m_Size) /
		m_YOLODeepNN->m_Layers[layerIdx].m_Stride + 1;

	m_YOLODeepNN->m_Layers[layerIdx].m_OutW = (m_YOLODeepNN->m_Layers[layerIdx].m_W + 2 * m_YOLODeepNN->m_Layers[layerIdx].m_Pad - m_YOLODeepNN->m_Layers[layerIdx].m_Size) /
		m_YOLODeepNN->m_Layers[layerIdx].m_Stride + 1;

	m_YOLODeepNN->m_Layers[layerIdx].m_OutC = m_YOLODeepNN->m_Layers[layerIdx].m_N;

	m_YOLODeepNN->m_Layers[layerIdx].m_Outputs = m_YOLODeepNN->m_Layers[layerIdx].m_OutH * m_YOLODeepNN->m_Layers[layerIdx].m_OutW * m_YOLODeepNN->m_Layers[layerIdx].m_OutC;

	m_YOLODeepNN->m_Layers[layerIdx].m_Inputs = m_YOLODeepNN->m_Layers[layerIdx].m_W * m_YOLODeepNN->m_Layers[layerIdx].m_H * m_YOLODeepNN->m_Layers[layerIdx].m_C;

	m_YOLODeepNN->m_Layers[layerIdx].m_Output = (float*)calloc(m_YOLODeepNN->m_Layers[layerIdx].m_Batch *  m_YOLODeepNN->m_Layers[layerIdx].m_Outputs, sizeof(float));

	if (m_YOLODeepNN->m_Layers[layerIdx].m_BatchNormalize) {

		m_YOLODeepNN->m_Layers[layerIdx].m_Scales = (float*)calloc(m_YOLODeepNN->m_Layers[layerIdx].m_N, sizeof(float));

		for (int i = 0; i < m_YOLODeepNN->m_Layers[layerIdx].m_N; ++i)
			m_YOLODeepNN->m_Layers[layerIdx].m_Scales[i] = 1;

		m_YOLODeepNN->m_Layers[layerIdx].m_RollingMean = (float*)calloc(m_YOLODeepNN->m_Layers[layerIdx].m_N, sizeof(float));
		m_YOLODeepNN->m_Layers[layerIdx].m_RollingVariance = (float*)calloc(m_YOLODeepNN->m_Layers[layerIdx].m_N, sizeof(float));
	}


	if (m_YOLODeepNN->m_GpuIndex >= 0) {

		m_YOLODeepNN->m_Layers[layerIdx].m_Weights_Gpu = m_OCLManager->InitializeFloatArray(m_YOLODeepNN->m_Layers[layerIdx].m_Weights, weightsLength);
		m_YOLODeepNN->m_Layers[layerIdx].m_Biases_Gpu = m_OCLManager->InitializeFloatArray(m_YOLODeepNN->m_Layers[layerIdx].m_Biases, m_YOLODeepNN->m_Layers[layerIdx].m_N);
		
		for(int i = 0; i < 2; i++) 
			m_YOLODeepNN->m_Layers[layerIdx].m_OutputSwapGPUBuffers[i] = m_OCLManager->InitializeFloatArray(m_YOLODeepNN->m_Layers[layerIdx].m_Output,
				m_YOLODeepNN->m_Layers[layerIdx].m_Batch * m_YOLODeepNN->m_Layers[layerIdx].m_OutH *
				m_YOLODeepNN->m_Layers[layerIdx].m_OutW * m_YOLODeepNN->m_Layers[layerIdx].m_N);
	}

	m_YOLODeepNN->m_Layers[layerIdx].m_Workspace_Size = m_YOLODeepNN->m_Layers[layerIdx].m_OutH * m_YOLODeepNN->m_Layers[layerIdx].m_OutW *
		m_YOLODeepNN->m_Layers[layerIdx].m_Size * m_YOLODeepNN->m_Layers[layerIdx].m_Size * m_YOLODeepNN->m_Layers[layerIdx].m_C * sizeof(float);
	m_YOLODeepNN->m_Layers[layerIdx].m_Activation = activation;

	printf("conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", 
		m_YOLODeepNN->m_Layers[layerIdx].m_N, 
		m_YOLODeepNN->m_Layers[layerIdx].m_Size, 
		m_YOLODeepNN->m_Layers[layerIdx].m_Size, 
		m_YOLODeepNN->m_Layers[layerIdx].m_Stride, 
		m_YOLODeepNN->m_Layers[layerIdx].m_W, 
		m_YOLODeepNN->m_Layers[layerIdx].m_H, 
		m_YOLODeepNN->m_Layers[layerIdx].m_C, 
		m_YOLODeepNN->m_Layers[layerIdx].m_OutW, 
		m_YOLODeepNN->m_Layers[layerIdx].m_OutH, 
		m_YOLODeepNN->m_Layers[layerIdx].m_OutC);

	return true;
}


bool YOLONeuralNet::PrepareRegionTypeLayer(int sectionIdx, int layerIdx, StructLayerFeedParams *layerFeedParams) {

	int coords = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "coords", 4);
	int classes = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "classes", 20);
	int num = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "num", 1);

	m_YOLODeepNN->m_Layers[layerIdx].m_LayerType = EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_REGION;
	m_YOLODeepNN->m_Layers[layerIdx].m_N = num;
	m_YOLODeepNN->m_Layers[layerIdx].m_Batch = layerFeedParams->m_Batch;
	m_YOLODeepNN->m_Layers[layerIdx].m_H = layerFeedParams->m_H;
	m_YOLODeepNN->m_Layers[layerIdx].m_W = layerFeedParams->m_W;
	m_YOLODeepNN->m_Layers[layerIdx].m_Classes = classes;
	m_YOLODeepNN->m_Layers[layerIdx].m_Coords = coords;
	m_YOLODeepNN->m_Layers[layerIdx].m_Biases = (float*)calloc(num * 2, sizeof(float));
	m_YOLODeepNN->m_Layers[layerIdx].m_Outputs = layerFeedParams->m_H * layerFeedParams->m_W * num * (classes + coords + 1);
	m_YOLODeepNN->m_Layers[layerIdx].m_Inputs = m_YOLODeepNN->m_Layers[layerIdx].m_Outputs;
	
	for (int i = 0; i < num * 2; ++i)
		m_YOLODeepNN->m_Layers[layerIdx].m_Biases[i] = .5;


#ifndef PINNED_MEM_OUTPUT
	m_YOLODeepNN->m_Layers[layerIdx].m_Output_Gpu = m_OCLManager->InitializeFloatArray(m_YOLODeepNN->m_Layers[layerIdx].m_Output, 
		m_YOLODeepNN->m_Layers[layerIdx].m_Batch * m_YOLODeepNN->m_Layers[layerIdx].m_Outputs);
#else

	m_YOLODeepNN->m_Layers[layerIdx].m_PinnedBuffer = m_OCLManager->InitializePinnedFloatArray(m_YOLODeepNN->m_Layers[layerIdx].m_Batch * m_YOLODeepNN->m_Layers[layerIdx].m_Outputs);
	m_YOLODeepNN->m_Layers[layerIdx].m_Output_Gpu = m_YOLODeepNN->m_Layers[layerIdx].m_PinnedBuffer->m_OCLBuffer;
	m_YOLODeepNN->m_Layers[layerIdx].m_PinnedOutput = (float*)m_YOLODeepNN->m_Layers[layerIdx].m_PinnedBuffer->m_PinnedMemory;
#endif

	printf("detection\n");
	srand(0);


	m_YOLODeepNN->m_Layers[layerIdx].m_ClassFix = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "classfix", 0);

	char outStr[512];
	strcpy(outStr, m_IniReader->GetValue((char*)m_LayerNames[sectionIdx].c_str(), "anchors", ""));

	char *a = outStr;

	if (a) {

		int len = (int)strlen(a);
		int n = 1;
		int i;
		for (i = 0; i < len; ++i) {

			if (a[i] == ',') 
				++n;
		}

		for (i = 0; i < n; ++i) {

			float bias = (float)atof(a);
			m_YOLODeepNN->m_Layers[layerIdx].m_Biases[i] = bias;
			a = strchr(a, ',') + 1;
		}
	}

	return true;
}

bool YOLONeuralNet::PrepareMaxpoolTypeLayer(int sectionIdx, int layerIdx, StructLayerFeedParams *layerFeedParams) {

	int stride = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "stride", 1);
	int size = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "size", stride);
	int padding = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[sectionIdx].c_str(), "padding", (size - 1) / 2);

	m_YOLODeepNN->m_Layers[layerIdx].m_LayerType = EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_MAXPOOL;
	m_YOLODeepNN->m_Layers[layerIdx].m_Batch = layerFeedParams->m_Batch;
	m_YOLODeepNN->m_Layers[layerIdx].m_H = layerFeedParams->m_H;
	m_YOLODeepNN->m_Layers[layerIdx].m_W = layerFeedParams->m_W;
	m_YOLODeepNN->m_Layers[layerIdx].m_C = layerFeedParams->m_C;
	m_YOLODeepNN->m_Layers[layerIdx].m_Pad = padding;
	m_YOLODeepNN->m_Layers[layerIdx].m_OutW = (m_YOLODeepNN->m_Layers[layerIdx].m_W + 2 * padding) / stride;
	m_YOLODeepNN->m_Layers[layerIdx].m_OutH = (m_YOLODeepNN->m_Layers[layerIdx].m_H + 2 * padding) / stride;
	m_YOLODeepNN->m_Layers[layerIdx].m_OutC = m_YOLODeepNN->m_Layers[layerIdx].m_C;
	m_YOLODeepNN->m_Layers[layerIdx].m_Outputs = m_YOLODeepNN->m_Layers[layerIdx].m_OutH * m_YOLODeepNN->m_Layers[layerIdx].m_OutW * m_YOLODeepNN->m_Layers[layerIdx].m_OutC;
	m_YOLODeepNN->m_Layers[layerIdx].m_Inputs = m_YOLODeepNN->m_Layers[layerIdx].m_H * m_YOLODeepNN->m_Layers[layerIdx].m_W * m_YOLODeepNN->m_Layers[layerIdx].m_C;
	m_YOLODeepNN->m_Layers[layerIdx].m_Size = size;
	m_YOLODeepNN->m_Layers[layerIdx].m_Stride = stride;
	int outSize = m_YOLODeepNN->m_Layers[layerIdx].m_OutH * m_YOLODeepNN->m_Layers[layerIdx].m_OutW * m_YOLODeepNN->m_Layers[layerIdx].m_OutC * m_YOLODeepNN->m_Layers[layerIdx].m_Batch;

	m_YOLODeepNN->m_Layers[layerIdx].m_Output_Gpu = m_OCLManager->InitializeFloatArray(NULL, outSize);


	printf("max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, 
						m_YOLODeepNN->m_Layers[layerIdx].m_W, 
						m_YOLODeepNN->m_Layers[layerIdx].m_H, 
						m_YOLODeepNN->m_Layers[layerIdx].m_C, 
						m_YOLODeepNN->m_Layers[layerIdx].m_OutW, 
						m_YOLODeepNN->m_Layers[layerIdx].m_OutH, 
						m_YOLODeepNN->m_Layers[layerIdx].m_OutC);
	return true;
}

bool YOLONeuralNet::ParseNNLayers() {

	EnumYOLODeepNNLayerType layerType;
	int layerCount = 0;
	size_t workspaceSize = 0;

	StructLayerFeedParams layerFeedParams;
	layerFeedParams.m_H = m_YOLODeepNN->m_H;
	layerFeedParams.m_W = m_YOLODeepNN->m_W;
	layerFeedParams.m_C = m_YOLODeepNN->m_C;
	layerFeedParams.m_Inputs = m_YOLODeepNN->m_Inputs;
	layerFeedParams.m_Batch = m_YOLODeepNN->m_BatchSize;

	for (int i = 1; i <= m_YOLODeepNN->m_TotalLayers; i++) {

		layerFeedParams.m_Index = layerCount;
		printf("%5d ", layerCount);
		layerType = MapNNLayerTypeStr(m_LayerNames[i]);

		switch (layerType) {
			
			case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_CONVOLUTIONAL:
				
				PrepareConvolutionalTypeLayer(i, layerCount, &layerFeedParams);
				break;

			case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_REGION:

				PrepareRegionTypeLayer(i, layerCount, &layerFeedParams);
				break;

			case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_MAXPOOL:

				PrepareMaxpoolTypeLayer(i, layerCount, &layerFeedParams);
				break;

			default:
				break;
		}

		
		m_YOLODeepNN->m_Layers[layerCount].m_DontLoad = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[i].c_str(), "dontload", 0);
		m_YOLODeepNN->m_Layers[layerCount].m_DontLoadScales = (int)m_IniReader->GetDoubleValue((char*)m_LayerNames[i].c_str(), "dontloadscales", 0);
		
		if (m_YOLODeepNN->m_Layers[layerCount].m_Workspace_Size > workspaceSize)
			workspaceSize = m_YOLODeepNN->m_Layers[layerCount].m_Workspace_Size;

		layerFeedParams.m_H = m_YOLODeepNN->m_Layers[layerCount].m_OutH;
		layerFeedParams.m_W = m_YOLODeepNN->m_Layers[layerCount].m_OutW;
		layerFeedParams.m_C = m_YOLODeepNN->m_Layers[layerCount].m_OutC;
		layerFeedParams.m_Inputs = m_YOLODeepNN->m_Layers[layerCount].m_Outputs;
		++layerCount;
	}

	if (workspaceSize)
		m_YOLODeepNN->m_Workspace = m_OCLManager->InitializeFloatArray(NULL, (workspaceSize - 1) / sizeof(float) + 1);

	return true;
}


//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO

void transpose_matrix(float *a, int rows, int cols) {

	float *transpose = (float*)calloc(rows*cols, sizeof(float));
	int x, y;
	for (x = 0; x < rows; ++x)
		for (y = 0; y < cols; ++y)
			transpose[y*rows + x] = a[x*cols + y];

	memcpy(a, transpose, rows*cols * sizeof(float));
	free(transpose);
}

bool YOLONeuralNet::ParseNNWeights() {

	FILE *fp = fopen(m_WeightsFile, "rb");
	if (!fp) {

		printf("ERROR - Failed to find NN weights file %s\n", m_WeightsFile);
		return false;
	}

	int majorRev;
	int minorRev;
	int revNum;
	int filterIdx;
	int totalExamples;
	StructYOLODeepNNLayer yoloDeepNNLayer;
	fread(&majorRev, sizeof(int), 1, fp);
	fread(&minorRev, sizeof(int), 1, fp);
	fread(&revNum, sizeof(int), 1, fp);
	fread(&totalExamples, sizeof(int), 1, fp);

	//int isTransposeEnabled = (majorRev > 1000) || (minorRev > 1000);

	for (int i = 0; i < m_YOLODeepNN->m_TotalLayers ; ++i) {

		yoloDeepNNLayer = m_YOLODeepNN->m_Layers[i];

		if (yoloDeepNNLayer.m_DontLoad)
			continue;

		if (yoloDeepNNLayer.m_LayerType == EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_CONVOLUTIONAL) {
			
			int numWeights = yoloDeepNNLayer.m_N * yoloDeepNNLayer.m_C * yoloDeepNNLayer.m_Size * yoloDeepNNLayer.m_Size;
			fread(yoloDeepNNLayer.m_Biases, sizeof(float), yoloDeepNNLayer.m_N, fp);

			if (yoloDeepNNLayer.m_BatchNormalize 
				&& (!yoloDeepNNLayer.m_DontLoadScales)) {

				fread(yoloDeepNNLayer.m_Scales, sizeof(float), yoloDeepNNLayer.m_N, fp);
				fread(yoloDeepNNLayer.m_RollingMean, sizeof(float), yoloDeepNNLayer.m_N, fp);
				fread(yoloDeepNNLayer.m_RollingVariance, sizeof(float), yoloDeepNNLayer.m_N, fp);
			}

			fread(yoloDeepNNLayer.m_Weights, sizeof(float), numWeights, fp);

			if (yoloDeepNNLayer.m_BatchNormalize
				&& (!yoloDeepNNLayer.m_DontLoadScales)) {

				//Nice trick to fold batch normalization layer into convolution layer
				//This saves good amount of processing power
				//https://github.com/hollance/Forge/blob/master/Examples/YOLO/yolo2metal.py
				//http://machinethink.net/blog/object-detection-with-yolo/

				for (int j = 0; j < yoloDeepNNLayer.m_N; j++) {

					yoloDeepNNLayer.m_Biases[j] = yoloDeepNNLayer.m_Biases[j] - (yoloDeepNNLayer.m_RollingMean[j] * yoloDeepNNLayer.m_Scales[j]
						/ sqrt(yoloDeepNNLayer.m_RollingVariance[j] + 0.0001f));
				}

				for (int j = 0; j < numWeights; j++) {

					filterIdx = j / (yoloDeepNNLayer.m_C * yoloDeepNNLayer.m_Size * yoloDeepNNLayer.m_Size);
					yoloDeepNNLayer.m_Weights[j] = yoloDeepNNLayer.m_Weights[j] * yoloDeepNNLayer.m_Scales[filterIdx]
						/ sqrt(yoloDeepNNLayer.m_RollingVariance[filterIdx] + 0.0001f);
				}
			}

			if (yoloDeepNNLayer.m_Flipped)
				transpose_matrix(yoloDeepNNLayer.m_Weights, yoloDeepNNLayer.m_C * yoloDeepNNLayer.m_Size * yoloDeepNNLayer.m_Size, yoloDeepNNLayer.m_N);


			if (m_YOLODeepNN->m_GpuIndex >= 0) {
			
				m_OCLManager->WriteFloatArray(yoloDeepNNLayer.m_Weights_Gpu, yoloDeepNNLayer.m_Weights, numWeights);
				m_OCLManager->WriteFloatArray(yoloDeepNNLayer.m_Biases_Gpu, yoloDeepNNLayer.m_Biases, yoloDeepNNLayer.m_N);

				//Free CPU memory. We wont be needing the arrays anymore
				free(yoloDeepNNLayer.m_Biases);
				yoloDeepNNLayer.m_Biases = NULL;

				free(yoloDeepNNLayer.m_RollingMean);
				yoloDeepNNLayer.m_RollingMean = NULL;

				free(yoloDeepNNLayer.m_RollingVariance);
				yoloDeepNNLayer.m_RollingVariance = NULL;

				free(yoloDeepNNLayer.m_Weights);
				yoloDeepNNLayer.m_Weights = NULL;
			}
		}
	}

	fclose(fp);
	return true;
}

bool YOLONeuralNet::Initialize() {

	std::ifstream classLabelsFile(m_ClassLabelsFile);

	if (!classLabelsFile.good()) {

		//LOG file doesnot exist
		return false;
	}

	std::copy(std::istream_iterator<std::string>(classLabelsFile),
		std::istream_iterator<std::string>(),
		std::back_inserter(m_ClassLabels));

	m_IniReader = new CSimpleIniA(false, false, false);
	m_IniReader->LoadFile(m_NetworkConfigFile);

	m_OCLManager = new OCLManager(m_GPUDevType);
	if (m_OCLManager->Initialize() != OCL_STATUS_READY) {
	
		//Log error
		logWriteFunc("Failed to initialize OCLManager object", EnumLogMsgType::LOG_MSG_TYPE_ERROR);
		return false;
	}

	strcpy(m_OCLDeviceName, m_OCLManager->GetDeviceName());


	m_YOLODeepNN = new StructYOLODeepNN;
	memset(m_YOLODeepNN, 0, sizeof(StructYOLODeepNN));
	ParseNetworkConfiguration();
	ParseNNLayers();
	ParseNNWeights();

	m_YOLODeepNN->m_BatchSize = 1;
	for(int i = 0; i < m_YOLODeepNN->m_TotalLayers; i++)
		m_YOLODeepNN->m_Layers[i].m_Batch = 1;

	srand(2222222);


	return true;
}

void YOLONeuralNet::Finalize() {

	if (m_YOLODeepNN->m_Workspace != NULL) {

		delete m_YOLODeepNN->m_Workspace;
		m_YOLODeepNN->m_Workspace = NULL;
	}

	for (int i = 0; i < m_YOLODeepNN->m_TotalLayers; i++) {

		switch(m_YOLODeepNN->m_Layers[i].m_LayerType){

		case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_CONVOLUTIONAL:

			m_OCLManager->FinalizeFloatArray(m_YOLODeepNN->m_Layers[i].m_Weights_Gpu);
			m_YOLODeepNN->m_Layers[i].m_Weights_Gpu = NULL;

			m_OCLManager->FinalizeFloatArray(m_YOLODeepNN->m_Layers[i].m_Biases_Gpu);
			m_YOLODeepNN->m_Layers[i].m_Biases_Gpu = NULL;

			for (int j = 0; j < 2; j++) {

				m_OCLManager->FinalizeFloatArray(m_YOLODeepNN->m_Layers[i].m_OutputSwapGPUBuffers[j]);
				m_YOLODeepNN->m_Layers[i].m_OutputSwapGPUBuffers[j] = NULL;
			}


			break;

		case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_MAXPOOL:

			m_OCLManager->FinalizeFloatArray(m_YOLODeepNN->m_Layers[i].m_Output_Gpu);
			m_YOLODeepNN->m_Layers[i].m_Output_Gpu = NULL;

			break;

		case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_REGION:

			free(m_YOLODeepNN->m_Layers[i].m_Biases);
			m_YOLODeepNN->m_Layers[i].m_Biases = NULL;

 			m_OCLManager->FinalizePinnedFloatArray(m_YOLODeepNN->m_Layers[i].m_PinnedBuffer);
			m_YOLODeepNN->m_Layers[i].m_PinnedBuffer = NULL;

			break;
		}
	}

	delete m_YOLODeepNN;
	delete m_IniReader;

	m_OCLManager->Finalize();
	delete m_OCLManager;
	m_ClassLabels.clear();
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
float overlap(float x1, float w1, float x2, float w2) {

	float l1 = x1 - w1 / 2;
	float l2 = x2 - w2 / 2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1 / 2;
	float r2 = x2 + w2 / 2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
float box_intersection(StructDetectionBBox a, StructDetectionBBox b) {

	float w = overlap(a.m_X, a.m_W, b.m_X, b.m_W);
	float h = overlap(a.m_Y, a.m_H, b.m_Y, b.m_H);
	if (w < 0 || h < 0) return 0;
	float area = w*h;
	return area;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
float box_union(StructDetectionBBox a, StructDetectionBBox b) {

	float i = box_intersection(a, b);
	float u = a.m_W * a.m_H + b.m_W * b.m_W - i;
	return u;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
float box_iou(StructDetectionBBox a, StructDetectionBBox b) {

	return box_intersection(a, b) / box_union(a, b);
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
int nms_comparator(const void *pa, const void *pb) {

	StructSortableBBox a = *(StructSortableBBox *)pa;
	StructSortableBBox b = *(StructSortableBBox *)pb;
	float diff = a.m_ProbScores[a.m_Index][b.m_ClassIdx] - b.m_ProbScores[b.m_Index][b.m_ClassIdx];
	if (diff < 0) return 1;
	else if (diff > 0) return -1;
	return 0;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
void YOLONeuralNet::ApplyNMS(StructDetectionBBox *boxes, float **probs, int total, int classes, float thresh) {

	StructSortableBBox *s = (StructSortableBBox*)calloc(total, sizeof(StructSortableBBox));

	for (int i = 0; i < total; ++i) {

		s[i].m_Index = i;
		s[i].m_ClassIdx = 0;
		s[i].m_ProbScores = probs;
	}

	for (int k = 0; k < classes; ++k) {
		for (int i = 0; i < total; ++i) {

			s[i].m_ClassIdx = k;
		}

		qsort(s, total, sizeof(StructSortableBBox), nms_comparator);

		for (int i = 0; i < total; ++i) {

			if (probs[s[i].m_Index][k] == 0) 
				continue;

			StructDetectionBBox a = boxes[s[i].m_Index];

			for (int j = i + 1; j < total; ++j) {

				StructDetectionBBox b = boxes[s[j].m_Index];
				if (box_iou(a, b) > thresh)
					probs[s[j].m_Index][k] = 0;
			}
		}
	}
	free(s);
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
StructDetectionBBox YOLONeuralNet::GetRegionBBox(float *x, float *biases, int n, int index, int i, int j, int w, int h) {

	StructDetectionBBox bBox;
	bBox.m_X = (i + LogisticActivate(x[index + 0])) / w;
	bBox.m_Y = (j + LogisticActivate(x[index + 1])) / h;
	bBox.m_W = exp(x[index + 2]) * biases[2 * n];
	bBox.m_H = exp(x[index + 3]) * biases[2 * n + 1];
	
	if (DOABS) {

		bBox.m_W = exp(x[index + 2]) * biases[2 * n] / w;
		bBox.m_H = exp(x[index + 3]) * biases[2 * n + 1] / h;
	}
	return bBox;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
void YOLONeuralNet::GetDetectionBBoxes(StructYOLODeepNNLayer *nnLayer, int w, int h, float thresh, float **probs, StructDetectionBBox *bBoxes, int onlyObjectness, int *map) {

	int j;
	//float *predictions = nnLayer->m_Output;
	float *predictions = NULL;
	
#ifndef PINNED_MEM_OUTPUT
	predictions = nnLayer->m_Output;
#else
	predictions = nnLayer->m_PinnedOutput;
#endif

	for (int i = 0; i < nnLayer->m_W * nnLayer->m_H; ++i) {

		int row = i / nnLayer->m_W;
		int col = i % nnLayer->m_W;
		
		for (int n = 0; n < nnLayer->m_N; ++n) {

			int index = i * nnLayer->m_N + n;
			int p_index = index * (nnLayer->m_Classes + 5) + 4;
			float scale = predictions[p_index];
			if (nnLayer->m_ClassFix == -1 && scale < .5) 
				scale = 0;

			int box_index = index * (nnLayer->m_Classes + 5);
			bBoxes[index] = GetRegionBBox(predictions, nnLayer->m_Biases, n, box_index, col, row, nnLayer->m_W, nnLayer->m_H);
			bBoxes[index].m_X *= w;
			bBoxes[index].m_Y *= h;
			bBoxes[index].m_W *= w;
			bBoxes[index].m_H *= h;

			int class_index = index * (nnLayer->m_Classes + 5) + 5;
			for (j = 0; j < nnLayer->m_Classes; ++j) {

				float prob = scale*predictions[class_index + j];
				probs[index][j] = (prob > thresh) ? prob : 0;
			}
			if (onlyObjectness) 
				probs[index][0] = scale;
		}
	}
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
void add_pixel(StructImage *m, int x, int y, int c, float val) {

	assert(x < m->m_W && y < m->m_H && c < m->m_C);
	m->m_DataArray[c*m->m_H*m->m_W + y*m->m_W + x] += val;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
void set_pixel(StructImage *m, int x, int y, int c, float val) {

	if (x < 0 || y < 0 || c < 0 || x >= m->m_W || y >= m->m_H || c >= m->m_C) return;
	assert(x < m->m_W && y < m->m_H && c < m->m_C);
	m->m_DataArray[c*m->m_H*m->m_W + y*m->m_W + x] = val;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
float get_pixel(StructImage *m, int x, int y, int c) {

	assert(x < m->m_W && y < m->m_H && c < m->m_C);
	return m->m_DataArray[c*m->m_H*m->m_W + y*m->m_W + x];
}


//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
float get_color(int c, int x, int max) {

	float ratio = ((float)x / max) * 5;
	int i = (int)floor(ratio);
	int j = (int)ceil(ratio);
	ratio -= i;
	float r = (1 - ratio) * BBOX_COLORS[i][c] + ratio*BBOX_COLORS[j][c];
	//printf("%f\n", r);
	return r;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
int max_index(float *a, int n) {

	if (n <= 0) return -1;
	int i, max_i = 0;
	float max = a[0];
	for (i = 1; i < n; ++i) {
		if (a[i] > max) {

			max = a[i];
			max_i = i;
		}
	}
	return max_i;
}

//Taken from https://github.com/pjreddie/darknet.git, Original author of YOLO
void DrawDetections(StructImage *im, int num, float thresh, StructDetectionBBox *boxes, float **probs, 
					std::vector<std::string> &names, int classes, cv::Mat &renderMat) { // * renderImage) {

	cv::Rect overlayRect;

	for (int i = 0; i < num; ++i) {
		
		int classidx = max_index(probs[i], classes);
		float prob = probs[i][classidx];

		if (prob > thresh) {

			int offset = classidx * 123457 % classes;
			float red = get_color(2, offset, classes);
			float green = get_color(1, offset, classes);
			float blue = get_color(0, offset, classes);
			StructDetectionBBox b = boxes[i];

			int left = (int)((b.m_X - b.m_W / 2.) * im->m_W);
			int right = (int)((b.m_X + b.m_W / 2.) * im->m_W);
			int top = (int)((b.m_Y - b.m_H / 2.) * im->m_H);
			int bot = (int)((b.m_Y + b.m_H / 2.) * im->m_H);
			 
			if (left < 0) left = 0;
			if (right > im->m_W - 1) right = im->m_W - 1;
			if (top < 0) top = 0;
			if (bot > im->m_H - 1) bot = im->m_H - 1;

			overlayRect.x = left;
			overlayRect.y = top;
			overlayRect.width = right - left;
			overlayRect.height = bot - top;

			cv::rectangle(renderMat, overlayRect, cv::Scalar(blue * 255, green * 255, red * 255), 2);
		}
	}
}

void YOLONeuralNet::PutCairoOverlay(
	StructRAWFrameSrcObject *srcRawFrameObject, 
	std::string const& timeText,
	cv::Point2d timeCenterPoint,
	std::string const& fontFace,
	double fontSize,
	cv::Scalar textColor,
	bool fontItalic,
	bool fontBold) {

	if (m_CairoSurface == NULL) {

		m_CairoSurface = cairo_image_surface_create(
			CAIRO_FORMAT_ARGB32,
			srcRawFrameObject->m_OverlayMat.cols,
			srcRawFrameObject->m_OverlayMat.rows);

		m_Cairo = cairo_create(m_CairoSurface);

		m_CairoTarget = cv::Mat(
			cairo_image_surface_get_height(m_CairoSurface),
			cairo_image_surface_get_width(m_CairoSurface),
			CV_8UC4,
			cairo_image_surface_get_data(m_CairoSurface),
			cairo_image_surface_get_stride(m_CairoSurface));
	}

	cv::cvtColor(srcRawFrameObject->m_OverlayMat, m_CairoTarget, cv::COLOR_BGR2BGRA);

	cairo_select_font_face(
		m_Cairo,
		fontFace.c_str(),
		fontItalic ? CAIRO_FONT_SLANT_ITALIC : CAIRO_FONT_SLANT_NORMAL,
		fontBold ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL);

	cairo_set_font_size(m_Cairo, fontSize);
	cairo_set_source_rgb(m_Cairo, textColor[2], textColor[1], textColor[0]);

	cairo_text_extents_t extents;
	cairo_text_extents(m_Cairo, timeText.c_str(), &extents);

	cairo_move_to(
		m_Cairo,
		timeCenterPoint.x - extents.width / 2 - extents.x_bearing,
		timeCenterPoint.y - extents.height / 2 - extents.y_bearing);

	cairo_show_text(m_Cairo, timeText.c_str());
	cv::cvtColor(m_CairoTarget, srcRawFrameObject->m_OverlayMat, cv::COLOR_BGRA2BGR);
}


int YOLONeuralNet::GetRemainingImagesCount() {

	return (int)m_ImageBatch.size();
}

void YOLONeuralNet::FetchNextImage(char *outImagePath, char *outImageName) {

	if (m_ImageBatch.size() > 0) {
		
		strcpy(outImagePath, m_ImageBatch[0].c_str());
		m_ImageBatch.erase(m_ImageBatch.begin());
		strcpy(outImageName, m_ImageNames[0].c_str());
		m_ImageNames.erase(m_ImageNames.begin());
	}
}

void YOLONeuralNet::SignalEOS() {

	m_VideoFileEOS = true;
}

void YOLONeuralNet::CopyVideoFileName(char *dstFilePath) {

	strcpy(dstFilePath, m_SrcVideoPath);
}

bool YOLONeuralNet::OpenVideoFile(cv::Mat &dstMat) {

	m_AVFormatContext = avformat_alloc_context();
	if (avformat_open_input(&m_AVFormatContext, m_SrcVideoPath, NULL, NULL) != 0) {

		printf("ERROR : avformat_open_input()  Failed to open video file %s\n", m_SrcVideoPath);
		return false;
	}

	if (avformat_find_stream_info(m_AVFormatContext, NULL) < 0) {
		
		printf("ERROR : avformat_find_stream_info()  Failed to open video file %s\n", m_SrcVideoPath);
		return false;
	}

	m_VideoStreamIdx = -1;
	for (unsigned int i = 0; i < m_AVFormatContext->nb_streams; i++) {

		if (m_AVFormatContext->streams[i]->codec->coder_type == AVMEDIA_TYPE_VIDEO) {

			m_VideoStreamIdx = i;
			break;
		}
	}

	if (m_VideoStreamIdx == -1) {

		printf("ERROR : AVMEDIA_TYPE_VIDEO  Failed to open video file %s\n", m_SrcVideoPath);
		return false;
	}

	m_AVCodecCtx = m_AVFormatContext->streams[m_VideoStreamIdx]->codec;

	m_AVCodec = avcodec_find_decoder(m_AVCodecCtx->codec_id);

	if (m_AVCodec == NULL) {

		printf("ERROR : Codec not found ! Failed to open video file %s\n", m_SrcVideoPath);
		return false;
	}

	if (avcodec_open2(m_AVCodecCtx, m_AVCodec, NULL) < 0) {
	
		printf("ERROR : Failed to open codec ! Failed to open video file %s\n", m_SrcVideoPath);
		return false;
	}

	m_AVFrame = av_frame_alloc();
	m_AVFrameRGB = av_frame_alloc();

	AVPixelFormat  pFormat = AV_PIX_FMT_BGR24;
	m_NumRGBbytes = avpicture_get_size(pFormat, m_AVCodecCtx->width, m_AVCodecCtx->height);
	m_AVRGBBuffer = (uint8_t *)av_malloc(m_NumRGBbytes * sizeof(uint8_t));
	avpicture_fill((AVPicture *)m_AVFrameRGB, m_AVRGBBuffer, pFormat, m_AVCodecCtx->width, m_AVCodecCtx->height);

	m_ImgConvertCtx = sws_getCachedContext(NULL, m_AVCodecCtx->width, m_AVCodecCtx->height, m_AVCodecCtx->pix_fmt,
		m_AVCodecCtx->width, m_AVCodecCtx->height, AV_PIX_FMT_BGR24, SWS_BICUBIC, NULL, NULL, NULL);


	dstMat = cv::Mat(m_AVCodecCtx->height, m_AVCodecCtx->width, CV_8UC3);

	m_FpsNum = m_AVFormatContext->streams[m_VideoStreamIdx]->r_frame_rate.den;
	m_FpsDen = m_AVFormatContext->streams[m_VideoStreamIdx]->r_frame_rate.num;

	printf("Opened video file successfully - Width %d, height %d\n", m_AVCodecCtx->width, m_AVCodecCtx->height);

	return true;
}

bool YOLONeuralNet::FetchNextFrameFromVideo(cv::Mat &dstMat) {

	AVPacket avPacket;
	int frameFinished;
	bool videoPktFound = false;
	int resValue = -1;

	while (!videoPktFound) {

		resValue = av_read_frame(m_AVFormatContext, &avPacket);
		if (resValue >= 0) {

			if (avPacket.stream_index == m_VideoStreamIdx) {

				videoPktFound = true;
				avcodec_decode_video2(m_AVCodecCtx, m_AVFrame, &frameFinished, &avPacket);

				if (frameFinished) {

					sws_scale(m_ImgConvertCtx, ((AVPicture*)m_AVFrame)->data, ((AVPicture*)m_AVFrame)->linesize, 0,
						m_AVCodecCtx->height, ((AVPicture *)m_AVFrameRGB)->data, ((AVPicture *)m_AVFrameRGB)->linesize);

					memcpy(dstMat.data, m_AVFrameRGB->data[0], m_NumRGBbytes);

				}
			}
		}
		else
			return false;

		av_free_packet(&avPacket);
	}

	return true;
}

void YOLONeuralNet::CloseVideoFile() {

	if (m_ImgConvertCtx != NULL) {

		sws_freeContext(m_ImgConvertCtx);
		m_ImgConvertCtx = NULL;
	}

	if (m_AVRGBBuffer != NULL) {

		av_free(m_AVRGBBuffer);
		m_AVRGBBuffer = NULL;
	}

	if (m_AVFrameRGB) {
	
		av_frame_free(&m_AVFrameRGB);
		m_AVFrameRGB = NULL;
	}

	if (m_AVFrame != NULL) {
	
		av_frame_free(&m_AVFrame);
		m_AVFrame = NULL;
	}

	if (m_AVCodec != NULL) {

		avcodec_close(m_AVCodecCtx);
		m_AVCodecCtx = NULL;
	}

	if (m_AVFormatContext != NULL) {

		avformat_close_input(&m_AVFormatContext);
		avformat_free_context(m_AVFormatContext);
		m_AVFormatContext = NULL;
	}
}

StructRAWFrameSrcObject* InitializeRAWFrameObject(char const* fileName, cv::Mat *srcMat, int dnnWidth, int dnnHeight) {

	StructRAWFrameSrcObject *srcRAWFrameObject = NULL;
	
	if (!std::string(fileName).empty()) {

		srcRAWFrameObject = new StructRAWFrameSrcObject;
		srcRAWFrameObject->m_CurrentImageMat = cv::imread(std::string(fileName));
		if (srcRAWFrameObject->m_CurrentImageMat.data == NULL) {

			fprintf(stderr, "Cannot load image \"%s\"\n", fileName);
			char buff[256];
			sprintf(buff, "echo %s >> bad.list", fileName);
			system(buff);
			delete srcRAWFrameObject;
			srcRAWFrameObject = NULL;
			return NULL;
		}
	}
	else {

		srcRAWFrameObject = new StructRAWFrameSrcObject;
		srcRAWFrameObject->m_CurrentImageMat = srcMat->clone();
	}

	unsigned char *data = (unsigned char *)srcRAWFrameObject->m_CurrentImageMat.data;
	int h = srcRAWFrameObject->m_CurrentImageMat.rows;
	int w = srcRAWFrameObject->m_CurrentImageMat.cols;
	int c = srcRAWFrameObject->m_CurrentImageMat.channels();
	int step = (int)srcRAWFrameObject->m_CurrentImageMat.step;

	srcRAWFrameObject->m_SrcImage = new StructImage;
	srcRAWFrameObject->m_SrcImage->m_DataArray = (float*)calloc(h*w*c, sizeof(float));

	int count = 0;
	srcRAWFrameObject->m_SrcImage->m_W = w;
	srcRAWFrameObject->m_SrcImage->m_H = h;
	srcRAWFrameObject->m_SrcImage->m_C = c;


	for (int k = 0; k < c; ++k) {
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				srcRAWFrameObject->m_SrcImage->m_DataArray[count++] = (float)(data[i*step + j*c + k] / 255.);
			}
		}
	}

	for (int i = 0; i < srcRAWFrameObject->m_SrcImage->m_W * srcRAWFrameObject->m_SrcImage->m_H; ++i) {

		float swap = srcRAWFrameObject->m_SrcImage->m_DataArray[i];
		srcRAWFrameObject->m_SrcImage->m_DataArray[i] = srcRAWFrameObject->m_SrcImage->m_DataArray[i + srcRAWFrameObject->m_SrcImage->m_W * srcRAWFrameObject->m_SrcImage->m_H * 2];
		srcRAWFrameObject->m_SrcImage->m_DataArray[i + srcRAWFrameObject->m_SrcImage->m_W * srcRAWFrameObject->m_SrcImage->m_H * 2] = swap;
	}

	srcRAWFrameObject->m_ResizedImage = new StructImage;
	srcRAWFrameObject->m_ResizedImage->m_H = dnnHeight;
	srcRAWFrameObject->m_ResizedImage->m_W = dnnWidth;
	srcRAWFrameObject->m_ResizedImage->m_C = srcRAWFrameObject->m_SrcImage->m_C;
	srcRAWFrameObject->m_ResizedImage->m_DataArray = (float*)calloc(srcRAWFrameObject->m_ResizedImage->m_H * srcRAWFrameObject->m_ResizedImage->m_W * srcRAWFrameObject->m_ResizedImage->m_C, 
		sizeof(float));

	srcRAWFrameObject->m_TempImage = new StructImage;
	srcRAWFrameObject->m_TempImage->m_H = srcRAWFrameObject->m_SrcImage->m_H;
	srcRAWFrameObject->m_TempImage->m_W = dnnWidth;
	srcRAWFrameObject->m_TempImage->m_C = srcRAWFrameObject->m_SrcImage->m_C;
	srcRAWFrameObject->m_TempImage->m_DataArray = (float*)calloc(srcRAWFrameObject->m_TempImage->m_H * srcRAWFrameObject->m_TempImage->m_W * srcRAWFrameObject->m_TempImage->m_C, 
		sizeof(float));

	int r, k;
	float w_scale = (float)(srcRAWFrameObject->m_SrcImage->m_W - 1) / (dnnWidth - 1);
	float h_scale = (float)(srcRAWFrameObject->m_SrcImage->m_H - 1) / (dnnHeight - 1);
	for (k = 0; k < srcRAWFrameObject->m_SrcImage->m_C; ++k) {
		for (r = 0; r < srcRAWFrameObject->m_SrcImage->m_H; ++r) {
			for (c = 0; c < dnnWidth; ++c) {
				float val = 0;
				if (c == dnnWidth - 1 || srcRAWFrameObject->m_SrcImage->m_W == 1) {
					val = get_pixel(srcRAWFrameObject->m_SrcImage, srcRAWFrameObject->m_SrcImage->m_W - 1, r, k);
				}
				else {
					float sx = c*w_scale;
					int ix = (int)sx;
					float dx = sx - ix;
					val = (1 - dx) * get_pixel(srcRAWFrameObject->m_SrcImage, ix, r, k) + dx * get_pixel(srcRAWFrameObject->m_SrcImage, ix + 1, r, k);
				}
				set_pixel(srcRAWFrameObject->m_TempImage, c, r, k, val);
			}
		}
	}
	for (k = 0; k < srcRAWFrameObject->m_SrcImage->m_C; ++k) {
		for (r = 0; r < dnnHeight; ++r) {
			float sy = r*h_scale;
			int iy = (int)sy;
			float dy = sy - iy;
			for (c = 0; c < dnnWidth; ++c) {
				float val = (1 - dy) * get_pixel(srcRAWFrameObject->m_TempImage, c, iy, k);
				set_pixel(srcRAWFrameObject->m_ResizedImage, c, r, k, val);
			}
			if (r == dnnHeight - 1 || srcRAWFrameObject->m_SrcImage->m_H == 1) continue;
			for (c = 0; c < dnnWidth; ++c) {
				float val = dy * get_pixel(srcRAWFrameObject->m_TempImage, c, iy + 1, k);
				add_pixel(srcRAWFrameObject->m_ResizedImage, c, r, k, val);
			}
		}
	}

	srcRAWFrameObject->m_OverlayMat = cv::Mat(cv::Size(srcRAWFrameObject->m_SrcImage->m_W, 50), CV_8UC3);
	srcRAWFrameObject->m_OverlayMat.setTo((cv::Scalar)0);
	srcRAWFrameObject->m_OverlayFinalMat = cv::Mat(cv::Size(srcRAWFrameObject->m_OverlayMat.cols, srcRAWFrameObject->m_OverlayMat.rows), CV_8UC3);
	srcRAWFrameObject->m_OverlayRect.x = 0;
	srcRAWFrameObject->m_OverlayRect.y = 0;
	srcRAWFrameObject->m_OverlayRect.width = srcRAWFrameObject->m_OverlayMat.cols;
	srcRAWFrameObject->m_OverlayRect.height = srcRAWFrameObject->m_OverlayMat.rows;
	srcRAWFrameObject->m_DisplayImageMat = cv::Mat(cv::Size(srcRAWFrameObject->m_SrcImage->m_W, srcRAWFrameObject->m_SrcImage->m_H), CV_8UC3);

	return srcRAWFrameObject;
}

void YOLONeuralNet::EnqueueRAWFrame(StructRAWFrameSrcObject *rawFrameObject) {

	m_SrcFrameQueueMutex.lock();
	m_SrcFrameQueue.push(rawFrameObject);
	m_SrcFrameQueueMutex.unlock();
}

void YOLONeuralNet::WaitForSync() {

	while (m_SrcFrameQueue.size() > MAX_FRAME_QUEUE_ITEMS)
		WaitMilliSecs(2);

	if (m_EnableDisplay) {

		while (m_SinkFrameQueue.size() > 0)
			WaitMilliSecs(2);
	}
}

void ProcessImages(YOLONeuralNet *yoloNNObj) {

	char imagePath[FILENAME_MAX];
	char imageName[FILENAME_MAX];

	StructRAWFrameSrcObject *newRAWFrameObject = NULL;
	while (yoloNNObj->GetRemainingImagesCount() > 0) {

		yoloNNObj->WaitForSync();
		yoloNNObj->FetchNextImage(imagePath, imageName);
		newRAWFrameObject = InitializeRAWFrameObject(imagePath, NULL, yoloNNObj->GetDNNWidth(), yoloNNObj->GetDNNHeight());
		if(newRAWFrameObject != NULL){

			newRAWFrameObject->m_SingletonSrcObject = false;
			strcpy(newRAWFrameObject->m_WorkingImageName, imageName);
			yoloNNObj->EnqueueRAWFrame(newRAWFrameObject);
		}
	}
}

void ProcessVideo(YOLONeuralNet *yoloNNObj) {

	char videoFilePath[FILENAME_MAX];
	cv::Mat videoFrame;
	int frameCount = 0;
	StructRAWFrameSrcObject *newRAWFrameObject = NULL;

	yoloNNObj->CopyVideoFileName(videoFilePath);

	printf("Opening video file %s\n", videoFilePath);

	if (!yoloNNObj->OpenVideoFile(videoFrame)) {

		yoloNNObj->CloseVideoFile();
		yoloNNObj->SignalEOS();
		return;
	}

	while (1) {

		if (!yoloNNObj->FetchNextFrameFromVideo(videoFrame))
			break;

		yoloNNObj->WaitForSync();

		newRAWFrameObject = InitializeRAWFrameObject("", &videoFrame, yoloNNObj->GetDNNWidth(), yoloNNObj->GetDNNHeight());
		if (newRAWFrameObject != NULL) {

			newRAWFrameObject->m_SingletonSrcObject = false;
			sprintf(newRAWFrameObject->m_WorkingImageName, "frame_%06d.jpg", frameCount);
			yoloNNObj->EnqueueRAWFrame(newRAWFrameObject);
		}

		frameCount++;
	}

	yoloNNObj->SignalEOS();
	yoloNNObj->CloseVideoFile();
	
	if(videoFrame.data != NULL)
		videoFrame.release();
}

void FinalizeRAWFrameObject(StructRAWFrameSrcObject *rawFrameObject) {

	if (rawFrameObject != NULL) {

		if (rawFrameObject->m_SrcImage != NULL) {

			free(rawFrameObject->m_SrcImage->m_DataArray);
			delete rawFrameObject->m_SrcImage;
			rawFrameObject->m_SrcImage = NULL;
		}

		if (rawFrameObject->m_TempImage != NULL) {

			free(rawFrameObject->m_TempImage->m_DataArray);
			delete rawFrameObject->m_TempImage;
			rawFrameObject->m_TempImage = NULL;
		}

		if (rawFrameObject->m_ResizedImage != NULL) {

			free(rawFrameObject->m_ResizedImage->m_DataArray);
			delete rawFrameObject->m_ResizedImage;
			rawFrameObject->m_ResizedImage = NULL;
		}

		if (rawFrameObject->m_CurrentImageMat.data != NULL)
			rawFrameObject->m_CurrentImageMat.release();

		if (rawFrameObject->m_OverlayMat.data != NULL)
			rawFrameObject->m_OverlayMat.release();

		if (rawFrameObject->m_OverlayFinalMat.data != NULL)
			rawFrameObject->m_OverlayFinalMat.release();

		if (rawFrameObject->m_DisplayImageMat.data != NULL)
			rawFrameObject->m_DisplayImageMat.release();

		delete rawFrameObject;
		rawFrameObject = NULL;
	}
}

void YOLONeuralNet::ProcessSinkFramesInSequence() {

	StructRAWFrameSinkObject *outSinkObject = NULL;
	bool initStatus = false;

	m_SinkThreadStatus = EnumThreadStatus::THREAD_STATUS_RUNNING;

	while (m_SinkActive) {
		
		if (m_SinkFrameQueue.size() > 0) {

			m_SinkFrameQueueMutex.lock();
			outSinkObject = m_SinkFrameQueue.front();
			m_SinkFrameQueue.pop();
			m_SinkFrameQueueMutex.unlock();

			if (m_SaveOPType == EnumSaveOPType::SAVE_OUTPUT_TYPE_VIDEO && !initStatus) {

				//Prepare Encoder
				if(InitializeSinkResources(outSinkObject->m_RAWSrcObject, GetFPSNum(), GetFPSDen()))
					initStatus = true;
				else {
					
					logWriteFunc("Failed to initialize sink resources", EnumLogMsgType::LOG_MSG_TYPE_ERROR);
					break;
				}
			}

			//Graphics overlay
			PostProcessDetections(outSinkObject);

			if (!outSinkObject->m_RAWSrcObject->m_SingletonSrcObject)
				FinalizeRAWFrameObject(outSinkObject->m_RAWSrcObject);

			delete outSinkObject;
			outSinkObject = NULL;
		}
		else
			WaitMilliSecs(2);
	}

	if(m_SaveOPType == EnumSaveOPType::SAVE_OUTPUT_TYPE_VIDEO) {

		logWriteFunc("Finalizing Sink Resources", EnumLogMsgType::LOG_MSG_TYPE_INFO);
		FinalizeSinkResources();
	}

	m_SinkThreadStatus = EnumThreadStatus::THREAD_STATUS_TERMINATED;
}


#ifdef WIN32

DWORD WINAPI ProcessBatchInput(LPVOID lpParameter) {

	YOLONeuralNet *yoloNNObj = (YOLONeuralNet*)lpParameter;
	ProcessImages(yoloNNObj);
	return 0;
}

DWORD WINAPI ProcessVideoInput(LPVOID lpParameter) {

	YOLONeuralNet *yoloNNObj = (YOLONeuralNet*)lpParameter;
	ProcessVideo(yoloNNObj);
	return 0;
}

DWORD WINAPI ProcessOutput(LPVOID lpParameter) {

	YOLONeuralNet *yoloNNObj = (YOLONeuralNet*)lpParameter;
	yoloNNObj->ProcessSinkFramesInSequence();
	return 0;
}


#elif __linux__

void* ProcessBatchInput(void *ptr) {
	
	YOLONeuralNet *yoloNNObj = (YOLONeuralNet*)ptr;
	ProcessImages(yoloNNObj);
}

void* ProcessVideoInput(void *ptr) {
	
	YOLONeuralNet *yoloNNObj = (YOLONeuralNet*)ptr;
	ProcessVideo(yoloNNObj);
}

void* ProcessOutput(void *ptr) {

	YOLONeuralNet *yoloNNObj = (YOLONeuralNet*)ptr;
	yoloNNObj->ProcessSinkFramesInSequence();
	return 0;
}

#endif

bool YOLONeuralNet::InitializeSinkResources(StructRAWFrameSrcObject *rawFrameSinkObject, int fpsNum, int fpsDen) {

	std::string outFileName = m_OutFolder + std::string("//InferenceOutput.avi");

	m_SinkFrameCount = 0;
	m_AVSinkCodec = avcodec_find_encoder(AV_CODEC_ID_H264); 
	m_AVSinkFormat = av_guess_format(NULL, outFileName.c_str(), NULL);
	m_AVSinkFormatContext = NULL;
	if (avformat_alloc_output_context2(&m_AVSinkFormatContext, m_AVSinkFormat, NULL, NULL) < 0) {

		logWriteFunc("Failed to allocate sink resources", EnumLogMsgType::LOG_MSG_TYPE_ERROR);
		return false;
	}

	m_AVSinkStream = avformat_new_stream(m_AVSinkFormatContext, m_AVSinkCodec);
	m_AVSinkStream->time_base.num = fpsNum;
	m_AVSinkStream->time_base.den = fpsDen;

	m_AVSinkStream->codec = avcodec_alloc_context3(m_AVSinkCodec);
	if (!m_AVSinkStream->codec) {
		
		logWriteFunc("Failed to allocate codec context", EnumLogMsgType::LOG_MSG_TYPE_ERROR);
		return false;
	}

	m_AVSinkStream->codec->bit_rate = 2100000;
	m_AVSinkStream->codec->width = rawFrameSinkObject->m_DisplayImageMat.cols;
	m_AVSinkStream->codec->height = rawFrameSinkObject->m_DisplayImageMat.rows;
	m_AVSinkStream->codec->time_base.num = fpsNum;
	m_AVSinkStream->codec->time_base.den = fpsDen;
	m_AVSinkStream->codec->gop_size = 10;
	m_AVSinkStream->codec->max_b_frames = 1;
	m_AVSinkStream->codec->qmin = 30;
	m_AVSinkStream->codec->qmax = 40;
	m_AVSinkStream->codec->pix_fmt = AV_PIX_FMT_YUV420P;

	av_opt_set(m_AVSinkStream->codec->priv_data, "preset", "ultrafast", 0);

	int retVal = avcodec_open2(m_AVSinkStream->codec, m_AVSinkCodec, NULL);
	if (retVal < 0) {

		logWriteFunc("Failed to open H.264 codec", EnumLogMsgType::LOG_MSG_TYPE_ERROR);
		return false;
	}

	if (avio_open2(&m_AVSinkFormatContext->pb, outFileName.c_str(), AVIO_FLAG_WRITE, NULL, NULL) < 0) {

		logWriteFunc("Failed to open output AVI file", EnumLogMsgType::LOG_MSG_TYPE_ERROR);
		return false;
	}

	retVal = avformat_write_header(m_AVSinkFormatContext, NULL);
	if (retVal < 0) {

		logWriteFunc("Failed to write AVI file header", EnumLogMsgType::LOG_MSG_TYPE_ERROR);
		return false;
	}
	m_AVIHeaderWritten = true;


	m_AVSinkFrame = av_frame_alloc();
	AVPixelFormat pFormat = AV_PIX_FMT_YUV420P;
	m_AVSinkFrame->width = m_AVSinkStream->codec->width;
	m_AVSinkFrame->height = m_AVSinkStream->codec->height;
	m_AVSinkFrame->format = pFormat;
	m_SinkCopyYUVBytes = avpicture_get_size(pFormat, m_AVSinkFrame->width, m_AVSinkFrame->height);
	m_SinkYUVBuffer = (uint8_t *)av_malloc(m_SinkCopyYUVBytes * sizeof(uint8_t));
	avpicture_fill((AVPicture *)m_AVSinkFrame, m_SinkYUVBuffer, pFormat, m_AVSinkFrame->width, m_AVSinkFrame->height);

	m_AVSinkRGBFrame = av_frame_alloc();
	pFormat = AV_PIX_FMT_BGR24;
	m_AVSinkRGBFrame->width = m_AVSinkStream->codec->width;
	m_AVSinkRGBFrame->height = m_AVSinkStream->codec->height;
	m_AVSinkRGBFrame->format = pFormat;
	m_SinkCopyRGBBytes = avpicture_get_size(pFormat, m_AVSinkRGBFrame->width, m_AVSinkRGBFrame->height);
	m_SinkRGBBuffer = (uint8_t *)av_malloc(m_SinkCopyRGBBytes * sizeof(uint8_t));
	avpicture_fill((AVPicture *)m_AVSinkRGBFrame, m_SinkRGBBuffer, pFormat, m_AVSinkRGBFrame->width, m_AVSinkRGBFrame->height);

	m_SinkConvertCtx = sws_getCachedContext(NULL, m_AVSinkRGBFrame->width, m_AVSinkRGBFrame->height, AV_PIX_FMT_BGR24,
		m_AVSinkFrame->width, m_AVSinkFrame->height, AV_PIX_FMT_YUV420P, SWS_BICUBIC, NULL, NULL, NULL);

	return true;
}

bool YOLONeuralNet::ProcessSinkFrame(StructRAWFrameSinkObject *rawSinkFrameObject) {


	if (m_SaveOPType == EnumSaveOPType::SAVE_OUTPUT_TYPE_JPEG) {

		char outJPEGfile[4*FILENAME_MAX];
		sprintf(outJPEGfile, "%s//frame_%06d.jpg", m_OutFolder, m_SinkFrameCount);
		cv::imwrite(outJPEGfile, rawSinkFrameObject->m_RAWSrcObject->m_DisplayImageMat);
	}
	else if(m_SaveOPType == EnumSaveOPType::SAVE_OUTPUT_TYPE_VIDEO){

		AVPacket avPacket;
		avPacket.data = NULL;
		avPacket.size = 0;
		int encodeResult = -1;
		m_AVSinkFrame->pts = m_SinkFrameCount;
		av_init_packet(&avPacket);

		memcpy(m_AVSinkRGBFrame->data[0], rawSinkFrameObject->m_RAWSrcObject->m_DisplayImageMat.data, m_SinkCopyRGBBytes);

		sws_scale(m_SinkConvertCtx, ((AVPicture*)m_AVSinkRGBFrame)->data, ((AVPicture*)m_AVSinkRGBFrame)->linesize, 0,
			m_AVSinkRGBFrame->height, ((AVPicture *)m_AVSinkFrame)->data, ((AVPicture *)m_AVSinkFrame)->linesize);

		if (avcodec_encode_video2(m_AVSinkStream->codec, &avPacket, m_AVSinkFrame, &encodeResult) < 0) {

			logWriteFunc("Failed to encode frame in AVI file", EnumLogMsgType::LOG_MSG_TYPE_ERROR);
			return false;
		}

		if (encodeResult) {

			if (avPacket.pts != AV_NOPTS_VALUE)
				avPacket.pts = av_rescale_q(avPacket.pts, m_AVSinkStream->codec->time_base, m_AVSinkFormatContext->streams[0]->codec->time_base);
			if (avPacket.dts != AV_NOPTS_VALUE)
				avPacket.dts = av_rescale_q(avPacket.dts, m_AVSinkStream->codec->time_base, m_AVSinkFormatContext->streams[0]->codec->time_base);
			if (avPacket.duration > 0)
				avPacket.duration = (int)av_rescale_q(avPacket.duration, m_AVSinkStream->codec->time_base, m_AVSinkFormatContext->streams[0]->codec->time_base);

			if (m_AVSinkStream->codec->coded_frame->key_frame)
				avPacket.flags |= AV_PKT_FLAG_KEY;

			av_interleaved_write_frame(m_AVSinkFormatContext, &avPacket);
			av_packet_unref(&avPacket);
		}
	}
	m_SinkFrameCount++;
}

void YOLONeuralNet::FinalizeSinkResources() {

	if (m_AVIHeaderWritten) {
	
		if (m_AVSinkFormatContext != NULL)
			av_write_trailer(m_AVSinkFormatContext);

		if (m_AVSinkFormatContext->pb != NULL)
			avio_close(m_AVSinkFormatContext->pb);
	}

	if(m_AVSinkStream->codec != NULL)
		avcodec_free_context(&m_AVSinkStream->codec);

	if (m_SinkConvertCtx != NULL) {

		sws_freeContext(m_SinkConvertCtx);
		m_SinkConvertCtx = NULL;
	}

	if (m_SinkRGBBuffer != NULL) {

		av_free(m_SinkRGBBuffer);
		m_SinkRGBBuffer = NULL;
	}

	if (m_AVSinkRGBFrame) {

		av_frame_free(&m_AVSinkRGBFrame);
		m_AVFrameRGB = NULL;
	}

	if (m_SinkYUVBuffer != NULL) {

		av_free(m_SinkYUVBuffer);
		m_SinkYUVBuffer = NULL;
	}

	if (m_AVSinkFrame != NULL) {

		av_frame_free(&m_AVSinkFrame);
		m_AVSinkFrame = NULL;
	}
}

void YOLONeuralNet::ProcessVideo(char *srcVideoPath) {

	float inferenceDuration = 0.0f;

	strcpy(m_SrcVideoPath, srcVideoPath);

#ifdef WIN32

	HANDLE procSinkThread = CreateThread(NULL, 0, ProcessOutput, (LPVOID)this, 0, NULL);
	HANDLE procSrcThread = CreateThread(NULL, 0, ProcessVideoInput, (LPVOID)this, 0, NULL);
	sprintf(m_OutFolder, "%s\\output", ExePath().c_str());
	CreateDirectory(m_OutFolder, NULL);

#elif __linux__

	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	void *pthreadStatus;

	strcpy(m_OutFolder, "output");
	const int dirErr = mkdir(m_OutFolder, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (dirErr == -1)
		printf("Error creating directory %s! \n", m_OutFolder);

	int iret = pthread_create(&m_ProcSinkThread, NULL, ProcessOutput, this);
	if (iret != 0) {
		
		logWriteFunc("Failed to create ProcessOutput thread", EnumLogMsgType::LOG_MSG_TYPE_ERROR);
		return;
	}

	iret = pthread_create(&m_ProcSrcThread, NULL, ProcessVideoInput, this);
	if (iret != 0) {

		logWriteFunc("Failed to create ProcessVideoInput thread", EnumLogMsgType::LOG_MSG_TYPE_ERROR);
		return;
	}


#endif

	sprintf(m_OverlayDeviceProp, "Device : %s", m_OCLDeviceName);
	StructYOLODeepNNLayer *finalLayer = &m_YOLODeepNN->m_Layers[m_YOLODeepNN->m_TotalLayers - 1];

	StructDetectionBBox *detBBoxes = (StructDetectionBBox*)calloc(finalLayer->m_W * finalLayer->m_H * finalLayer->m_N, sizeof(StructDetectionBBox));
	float **detProbScores = (float**)calloc(finalLayer->m_W * finalLayer->m_H * finalLayer->m_N, sizeof(float *));

	for (int j = 0; j < finalLayer->m_W * finalLayer->m_H * finalLayer->m_N; ++j)
		detProbScores[j] = (float*)calloc(finalLayer->m_Classes, sizeof(float));

	int inputSize = m_YOLODeepNN->m_Layers[0].m_Inputs * m_YOLODeepNN->m_BatchSize;

	m_YoloNNCurrentState = new StructYOLODeepNNState;
	memset(m_YoloNNCurrentState, 0, sizeof(StructYOLODeepNNState));

	StructRAWFrameSrcObject *rawFrameObject = NULL;

	while (!m_VideoFileEOS) {

		if (m_SrcFrameQueue.size() > 0) {

			m_SrcFrameQueueMutex.lock();
			rawFrameObject = m_SrcFrameQueue.front();
			m_SrcFrameQueue.pop();
			m_SrcFrameQueueMutex.unlock();

			if (m_YoloNNCurrentState->m_InputRefGpu == NULL)
				m_YoloNNCurrentState->m_InputRefGpu = m_OCLManager->InitializeFloatArray(rawFrameObject->m_ResizedImage->m_DataArray, inputSize);
			else
				m_OCLManager->WriteFloatArray(m_YoloNNCurrentState->m_InputRefGpu, rawFrameObject->m_ResizedImage->m_DataArray, inputSize);

			RunInference(rawFrameObject, inferenceDuration, detProbScores, detBBoxes);
		}
		else
			WaitMilliSecs(2);
	}


#ifdef __linux__

	pthread_join(m_ProcSrcThread, &pthreadStatus);
	pthread_join(m_ProcSinkThread, &pthreadStatus);
#endif

	//Wait for sink thread to finish.
	while (m_SinkFrameQueue.size() > 0)
		WaitMilliSecs(2);

	m_SinkActive = false;

	while (m_SinkThreadStatus != EnumThreadStatus::THREAD_STATUS_TERMINATED)
		WaitMilliSecs(2);

	if (m_YoloNNCurrentState->m_InputRefGpu != NULL) {

		m_OCLManager->FinalizeFloatArray(m_YoloNNCurrentState->m_InputRefGpu);
		m_YoloNNCurrentState->m_InputRefGpu = NULL;
	}
	delete m_YoloNNCurrentState;
	m_YoloNNCurrentState = NULL;

	free(detBBoxes);

	for (int i = 0; i < finalLayer->m_W * finalLayer->m_H * finalLayer->m_N; i++)
		free(detProbScores[i]);

	cvWaitKey(0);
	cvDestroyAllWindows();
}

void YOLONeuralNet::ProcessImageBatch(char *srcFolder) {

	logWriteFunc("Enumerating image files in directory....", EnumLogMsgType::LOG_MSG_TYPE_INFO);

	EnumerateFilesInDirectory(string(srcFolder), m_ImageBatch, m_ImageNames);
	m_FpsNum = 1;
	m_FpsDen = 25;


#ifdef WIN32

	sprintf(m_OutFolder, "%s\\output", ExePath().c_str());
	CreateDirectory(m_OutFolder, NULL);
	HANDLE procSinkThread = CreateThread(NULL, 0, ProcessOutput, (LPVOID)this, 0, NULL);
	HANDLE procSrcThread = CreateThread(NULL, 0, ProcessBatchInput, (LPVOID)this, 0, NULL);


#elif __linux__

	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	void *pthreadStatus;

	strcpy(m_OutFolder, "output");
	mkdir(m_OutFolder, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	int iret = pthread_create(&m_ProcSinkThread, NULL, ProcessOutput, this);
	if (iret != 0) {

		logWriteFunc("Failed to create ProcessOutput thread", EnumLogMsgType::LOG_MSG_TYPE_ERROR);
		return;
	}

	iret = pthread_create(&m_ProcSrcThread, NULL, ProcessBatchInput, this);
	if (iret != 0) {

		logWriteFunc("Failed to create ProcessBatchInput thread", EnumLogMsgType::LOG_MSG_TYPE_ERROR);
		return;
	}
	

#endif

	int totalImages = GetRemainingImagesCount();

	if (totalImages == 0) {

		sprintf(m_LogMsgStr, "Failed to enumerate any images in folder %s.Terminating...", srcFolder);
		logWriteFunc(m_LogMsgStr, EnumLogMsgType::LOG_MSG_TYPE_ERROR);
		return;
	}
	else {

		sprintf(m_LogMsgStr, "Processing %d images in sequence.", totalImages);
		logWriteFunc(m_LogMsgStr, EnumLogMsgType::LOG_MSG_TYPE_INFO);
	}


	float inferenceDuration = 0.0f;

	sprintf(m_OverlayDeviceProp, "Device : %s", m_OCLDeviceName);
	StructYOLODeepNNLayer *finalLayer = &m_YOLODeepNN->m_Layers[m_YOLODeepNN->m_TotalLayers - 1];

	StructDetectionBBox *detBBoxes = (StructDetectionBBox*)calloc(finalLayer->m_W * finalLayer->m_H * finalLayer->m_N, sizeof(StructDetectionBBox));
	float **detProbScores = (float**)calloc(finalLayer->m_W * finalLayer->m_H * finalLayer->m_N, sizeof(float *));

	for (int j = 0; j < finalLayer->m_W * finalLayer->m_H * finalLayer->m_N; ++j)
		detProbScores[j] = (float*)calloc(finalLayer->m_Classes, sizeof(float));

	int inputSize = m_YOLODeepNN->m_Layers[0].m_Inputs * m_YOLODeepNN->m_BatchSize;

	m_YoloNNCurrentState = new StructYOLODeepNNState;
	memset(m_YoloNNCurrentState, 0, sizeof(StructYOLODeepNNState));

	StructRAWFrameSrcObject *rawFrameObject = NULL;

	logWriteFunc("Processing image files....", EnumLogMsgType::LOG_MSG_TYPE_INFO);

	while (GetRemainingImagesCount() > 0
		&& m_SinkThreadStatus == EnumThreadStatus::THREAD_STATUS_RUNNING) {

		if (m_SrcFrameQueue.size() > 0) {

			m_SrcFrameQueueMutex.lock();
			rawFrameObject = m_SrcFrameQueue.front();
			m_SrcFrameQueue.pop();
			m_SrcFrameQueueMutex.unlock();

			if (m_YoloNNCurrentState->m_InputRefGpu == NULL)
				m_YoloNNCurrentState->m_InputRefGpu = m_OCLManager->InitializeFloatArray(rawFrameObject->m_ResizedImage->m_DataArray, inputSize);
			else
				m_OCLManager->WriteFloatArray(m_YoloNNCurrentState->m_InputRefGpu, rawFrameObject->m_ResizedImage->m_DataArray, inputSize);

			RunInference(rawFrameObject, inferenceDuration, detProbScores, detBBoxes);
		}
		else
			WaitMilliSecs(2);
	}


#ifdef __linux__

	pthread_join(m_ProcSrcThread, &pthreadStatus);
	pthread_join(m_ProcSinkThread, &pthreadStatus);
#endif

	//Wait for sink thread to finish.
	while (m_SinkFrameQueue.size() > 0)
		WaitMilliSecs(2);

	m_SinkActive = false;

	while (m_SinkThreadStatus != EnumThreadStatus::THREAD_STATUS_TERMINATED)
		WaitMilliSecs(2);

	if (m_YoloNNCurrentState->m_InputRefGpu != NULL) {

		m_OCLManager->FinalizeFloatArray(m_YoloNNCurrentState->m_InputRefGpu);
		m_YoloNNCurrentState->m_InputRefGpu = NULL;
	}
	delete m_YoloNNCurrentState;
	m_YoloNNCurrentState = NULL;

	free(detBBoxes);

	for (int i = 0; i < finalLayer->m_W * finalLayer->m_H * finalLayer->m_N; i++)
		free(detProbScores[i]);

	cvWaitKey(0);
	cvDestroyAllWindows();
}

void YOLONeuralNet::ProcessSingleImage(char* inputFile) {

	char fileName[256];
	int BURN_ITERATIONS = 50;
	int iterIdx = 0;
	int NUM_ITERATIONS = (ENABLE_DISPLAY == 1) ? 5 : 1000;
	std::vector<float> timings;
	float inferenceDuration;
	m_FpsNum = 1;
	m_FpsDen = 25;

	m_SinkThreadStatus = EnumThreadStatus::THREAD_STATUS_INITIALIZED;

#ifdef WIN32

	sprintf(m_OutFolder, "%s\\output", ExePath().c_str());
	CreateDirectory(m_OutFolder, NULL);
	
	HANDLE procThread = CreateThread(NULL, 0, ProcessOutput, (LPVOID)this, 0, NULL);

#elif __linux__

	strcpy(m_OutFolder, "output");
	const int dirErr = mkdir(m_OutFolder, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (dirErr == -1)
		printf("Error creating directory %s! \n", m_OutFolder);

	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	void *pthreadStatus;

	//for(int i = 0; i < m_ImageBatch.size(); i++)
	//	printf("Image path : %s name %s\n", m_ImageBatch[i].c_str(), m_ImageNames[i].c_str());

	int iret = pthread_create(&m_ProcSrcThread, NULL, ProcessOutput, this);
	if (iret != 0) {

		logWriteFunc("Failed to create ProcessOutput thread", EnumLogMsgType::LOG_MSG_TYPE_ERROR);
		return;
	}


#endif

	sprintf(fileName, inputFile);

	StructRAWFrameSrcObject *rawFrameSrcObject = InitializeRAWFrameObject(fileName, NULL, m_YOLODeepNN->m_W, m_YOLODeepNN->m_H);
	if(rawFrameSrcObject == NULL){

		sprintf(m_LogMsgStr, "Failed to initialize RAW frame object for file %s", fileName);
		logWriteFunc(m_LogMsgStr, EnumLogMsgType::LOG_MSG_TYPE_ERROR);
		return;
	}

	rawFrameSrcObject->m_SingletonSrcObject = true;

	StructYOLODeepNNLayer *finalLayer = &m_YOLODeepNN->m_Layers[m_YOLODeepNN->m_TotalLayers - 1];
	StructDetectionBBox *detBBoxes = (StructDetectionBBox*)calloc(finalLayer->m_W * finalLayer->m_H * finalLayer->m_N, sizeof(StructDetectionBBox));
	float **detProbScores = (float**)calloc(finalLayer->m_W * finalLayer->m_H * finalLayer->m_N, sizeof(float *));

	for (int j = 0; j < finalLayer->m_W * finalLayer->m_H * finalLayer->m_N; ++j)
		detProbScores[j] = (float*)calloc(finalLayer->m_Classes, sizeof(float));

	sprintf(m_OverlayDeviceProp, "Device : %s", m_OCLDeviceName);

	int inputSize = m_YOLODeepNN->m_Layers[0].m_Inputs * m_YOLODeepNN->m_BatchSize;

	m_YoloNNCurrentState = new StructYOLODeepNNState;
	memset(m_YoloNNCurrentState, 0, sizeof(StructYOLODeepNNState));
	m_YoloNNCurrentState->m_InputRefGpu = m_OCLManager->InitializeFloatArray(rawFrameSrcObject->m_ResizedImage->m_DataArray, inputSize);

	for (int i = 0; i < NUM_ITERATIONS; i++) {

		sprintf(rawFrameSrcObject->m_WorkingImageName, "frame_%06d.jpg", iterIdx);
		RunInference(rawFrameSrcObject, inferenceDuration, detProbScores, detBBoxes);
		timings.push_back(inferenceDuration);
		iterIdx++;
	}

	if (NUM_ITERATIONS > BURN_ITERATIONS) {
	
		float avgSpeed = (float)std::accumulate(timings.begin() + BURN_ITERATIONS, timings.end(), 0.0) / (timings.size() - BURN_ITERATIONS);
		sprintf(m_LogMsgStr, "YoloOCLInference DNN Avg Proc Speed{ Time, FPS } : {%f, %f}", avgSpeed, 1000 / avgSpeed);
		logWriteFunc(m_LogMsgStr, EnumLogMsgType::LOG_MSG_TYPE_INFO);
	}

#ifdef __linux__

	pthread_join(m_ProcSrcThread, &pthreadStatus);
#endif

	//Wait for sink thread to finish.
	while (m_SinkFrameQueue.size() > 0)
		WaitMilliSecs(2);

	m_SinkActive = false;

	while(m_SinkThreadStatus != EnumThreadStatus::THREAD_STATUS_TERMINATED)
		WaitMilliSecs(2);

	if (m_YoloNNCurrentState->m_InputRefGpu != NULL) {

		m_OCLManager->FinalizeFloatArray(m_YoloNNCurrentState->m_InputRefGpu);
		m_YoloNNCurrentState->m_InputRefGpu = NULL;
	}
	delete m_YoloNNCurrentState;
	m_YoloNNCurrentState = NULL;

	FinalizeRAWFrameObject(rawFrameSrcObject);

	free(detBBoxes);

	for (int i = 0; i < finalLayer->m_W * finalLayer->m_H * finalLayer->m_N; i++)
		free(detProbScores[i]);

	cvWaitKey(0);
	cvDestroyAllWindows();
	logWriteFunc("Completed Processing single image input", EnumLogMsgType::LOG_MSG_TYPE_INFO);
}

void YOLONeuralNet::RunInference(StructRAWFrameSrcObject *rawFrameObject, float &inferenceDuration, float **detProbScores,
	StructDetectionBBox *detBBoxes) {

	StructYOLODeepNNLayer *finalLayer = &m_YOLODeepNN->m_Layers[m_YOLODeepNN->m_TotalLayers - 1];
	const auto start_time = std::chrono::steady_clock::now();

	m_YoloNNCurrentState->m_LayerIndex = 0;
	m_YoloNNCurrentState->m_DeepNN = m_YOLODeepNN;
	m_YoloNNCurrentState->m_InputGpu = m_YoloNNCurrentState->m_InputRefGpu;
	m_YoloNNCurrentState->m_Workspace = m_YOLODeepNN->m_Workspace;
	m_YoloNNCurrentState->m_ConvSwapBufIdx = (m_YoloNNCurrentState->m_ConvSwapBufIdx == 0) ? 1 : 0;

	for (int i = 0; i < m_YOLODeepNN->m_TotalLayers; ++i) {

		m_YoloNNCurrentState->m_LayerIndex = i;

		PropagateLayerInputsForward(&m_YOLODeepNN->m_Layers[i], m_YoloNNCurrentState);
		if (m_YOLODeepNN->m_Layers[i].m_LayerType == EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_CONVOLUTIONAL)
			m_YoloNNCurrentState->m_InputGpu = m_YOLODeepNN->m_Layers[i].m_OutputSwapGPUBuffers[m_YoloNNCurrentState->m_ConvSwapBufIdx];
		else
			m_YoloNNCurrentState->m_InputGpu = m_YOLODeepNN->m_Layers[i].m_Output_Gpu;
		m_YoloNNCurrentState->m_InputSize = m_YOLODeepNN->m_Layers[i].m_Batch * m_YOLODeepNN->m_Layers[i].m_OutH *
			m_YOLODeepNN->m_Layers[i].m_OutW * m_YOLODeepNN->m_Layers[i].m_N;
	}


	const auto elapsed_time = std::chrono::steady_clock::now() - start_time;
	inferenceDuration = std::chrono::duration<float, std::milli>(elapsed_time).count();

	sprintf(m_LogMsgStr, "Predicted in %2.2f ms @ %2.2f FPS", inferenceDuration, 1000 / inferenceDuration);
	logWriteFunc(m_LogMsgStr, EnumLogMsgType::LOG_MSG_TYPE_INFO);

	StructRAWFrameSinkObject *rawFrameSinkObject = new StructRAWFrameSinkObject;
	rawFrameSinkObject->m_RAWSrcObject = rawFrameObject;
	rawFrameSinkObject->m_DetBBoxes = detBBoxes;
	rawFrameSinkObject->m_DetProbScores = detProbScores;
	rawFrameSinkObject->m_InferenceDuration = inferenceDuration;
	rawFrameSinkObject->m_FinalLayer = finalLayer;

	m_SinkFrameQueueMutex.lock();
	m_SinkFrameQueue.push(rawFrameSinkObject);
	m_SinkFrameQueueMutex.unlock();

	cvWaitKey(1);

	if (m_EnableDisplay) {

		while(m_SinkFrameQueue.size() > 0)
			WaitMilliSecs(2);
	}
}

void YOLONeuralNet::PostProcessDetections(StructRAWFrameSinkObject *rawFrameSinkObject) {

	char overlayText[256];
	GetDetectionBBoxes(rawFrameSinkObject->m_FinalLayer, 1, 1, m_DetThreshold, rawFrameSinkObject->m_DetProbScores, rawFrameSinkObject->m_DetBBoxes, 0, 0);
	ApplyNMS(rawFrameSinkObject->m_DetBBoxes, rawFrameSinkObject->m_DetProbScores, 
		rawFrameSinkObject->m_FinalLayer->m_W * rawFrameSinkObject->m_FinalLayer->m_H * rawFrameSinkObject->m_FinalLayer->m_N, rawFrameSinkObject->m_FinalLayer->m_Classes, m_NMSOverlap);

	DrawDetections(rawFrameSinkObject->m_RAWSrcObject->m_SrcImage, rawFrameSinkObject->m_FinalLayer->m_W * rawFrameSinkObject->m_FinalLayer->m_H * rawFrameSinkObject->m_FinalLayer->m_N, m_DetThreshold,
		rawFrameSinkObject->m_DetBBoxes, rawFrameSinkObject->m_DetProbScores, m_ClassLabels, rawFrameSinkObject->m_FinalLayer->m_Classes, rawFrameSinkObject->m_RAWSrcObject->m_CurrentImageMat);

	rawFrameSinkObject->m_RAWSrcObject->m_OverlayMat.setTo((cv::Scalar)0);
	sprintf(overlayText, "Inference Duration : %2.2f ms Speed : %2.2f fps", rawFrameSinkObject->m_InferenceDuration, 1000 / rawFrameSinkObject->m_InferenceDuration);
	PutCairoOverlay(rawFrameSinkObject->m_RAWSrcObject, overlayText, cv::Point2d(180, 20), "arial", 15, cv::Scalar(0, 255, 255), false, true);
	PutCairoOverlay(rawFrameSinkObject->m_RAWSrcObject, m_OverlayDeviceProp, cv::Point2d(180, 40), "arial", 15, cv::Scalar(0, 255, 255), false, true);

	rawFrameSinkObject->m_RAWSrcObject->m_DisplayImageMat = rawFrameSinkObject->m_RAWSrcObject->m_CurrentImageMat.clone();
	cv::addWeighted(rawFrameSinkObject->m_RAWSrcObject->m_OverlayMat, 1, rawFrameSinkObject->m_RAWSrcObject->m_DisplayImageMat(rawFrameSinkObject->m_RAWSrcObject->m_OverlayRect), 
		0.5, 0.0, rawFrameSinkObject->m_RAWSrcObject->m_OverlayFinalMat);
	rawFrameSinkObject->m_RAWSrcObject->m_OverlayFinalMat += 0.4 * rawFrameSinkObject->m_RAWSrcObject->m_OverlayFinalMat;
	rawFrameSinkObject->m_RAWSrcObject->m_OverlayFinalMat.copyTo(rawFrameSinkObject->m_RAWSrcObject->m_DisplayImageMat(rawFrameSinkObject->m_RAWSrcObject->m_OverlayRect));


	if (m_SaveOPType != EnumSaveOPType::SAVE_OUTPUT_TYPE_NONE)
		ProcessSinkFrame(rawFrameSinkObject);

	if (m_EnableDisplay) {

		cv::imshow("Detections", rawFrameSinkObject->m_RAWSrcObject->m_DisplayImageMat);
		cvWaitKey(0);
	}
}

void PrintOCLBuffer(OCLBuffer *inBuffer, OCLManager *oclManager, char* fileName, int numItems) {

	
	float *verfArray = (float*)calloc(numItems, sizeof(float));
	oclManager->ReadFloatArray(verfArray, inBuffer, numItems);
	
	std::ofstream myfile(fileName);
	if (myfile.is_open()) {

		for (int count = 0; count < numItems; count++)
			myfile << verfArray[count] << "\n";
		myfile.close();
	}
	free(verfArray);
}

float YOLONeuralNet::PropagateLayerInputsForward(StructYOLODeepNNLayer *inLayer, StructYOLODeepNNState *netState) {

	int m = 0;
	int k = 0;
	int n = 0;
	int size = 0;
	int index = 0;
	int swapIdx = 0;
	int arrayLen = 0;
	
	float timeAccumulator = 0.0f;
	//char debugFileName[256];
	//const auto start_time = std::chrono::steady_clock::now();

	switch (inLayer->m_LayerType) {

		case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_CONVOLUTIONAL:

			m = inLayer->m_N;
			k = inLayer->m_Size * inLayer->m_Size * inLayer->m_C;
			n = inLayer->m_OutW * inLayer->m_OutH;

			timeAccumulator += m_OCLManager->ConvertImageToColumnArray(netState->m_InputGpu, inLayer->m_C, inLayer->m_H,
				inLayer->m_W, inLayer->m_Size, inLayer->m_Stride, inLayer->m_Pad, netState->m_Workspace);

			timeAccumulator += m_OCLManager->ComputeGEMM(false, false, m, n, k, 1.0f, inLayer->m_Weights_Gpu, 0, k,
								netState->m_Workspace, 0, n, 1.0f, inLayer->m_OutputSwapGPUBuffers[netState->m_ConvSwapBufIdx], 0, n);

			timeAccumulator += m_OCLManager->AddBias(inLayer->m_OutputSwapGPUBuffers[netState->m_ConvSwapBufIdx], inLayer->m_Biases_Gpu, inLayer->m_Batch,
				inLayer->m_N, inLayer->m_OutH * inLayer->m_OutW);

			swapIdx = (netState->m_ConvSwapBufIdx == 0) ? 1 : 0;

			timeAccumulator += m_OCLManager->ActivateInputs(inLayer->m_OutputSwapGPUBuffers[netState->m_ConvSwapBufIdx],
				inLayer->m_OutputSwapGPUBuffers[swapIdx], inLayer->m_Outputs * inLayer->m_Batch, inLayer->m_Activation);

			break;

		case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_REGION:

			timeAccumulator += m_OCLManager->FlattenArray(netState->m_InputGpu, inLayer->m_H * inLayer->m_W,
				inLayer->m_N * (inLayer->m_Coords + inLayer->m_Classes + 1), inLayer->m_Batch, 1, inLayer->m_Output_Gpu);


			timeAccumulator += m_OCLManager->SoftMax(inLayer->m_Output_Gpu, inLayer->m_Classes, inLayer->m_Classes + 5,
							inLayer->m_W * inLayer->m_H * inLayer->m_N * inLayer->m_Batch, 1, inLayer->m_Output_Gpu, 5);

#ifndef PINNED_MEM_OUTPUT
			m_OCLManager->ReadFloatArray(inLayer->m_Output, inLayer->m_Output_Gpu, inLayer->m_Batch * inLayer->m_Outputs);
#else
			m_OCLManager->ReadFloatArray(inLayer->m_PinnedOutput, inLayer->m_Output_Gpu, inLayer->m_Batch * inLayer->m_Outputs);
#endif

			size = inLayer->m_Coords + inLayer->m_Classes + 1;

			arrayLen = inLayer->m_H * inLayer->m_W * inLayer->m_N;
			
			//#pragma omp parallel num_threads(inLayer->m_N)	
			for (int i = 0; i < arrayLen; ++i) {

				index = size * i;
#ifndef PINNED_MEM_OUTPUT
				inLayer->m_Output[index + 4] = LogisticActivate(inLayer->m_Output[index + 4]);
#else
				inLayer->m_PinnedOutput[index + 4] = LogisticActivate(inLayer->m_PinnedOutput[index + 4]);
#endif
			}

			break;

		case EnumYOLODeepNNLayerType::YOLO_DNN_LAYER_MAXPOOL:

			n = inLayer->m_OutH * inLayer->m_OutW * inLayer->m_C * inLayer->m_Batch;
			timeAccumulator += m_OCLManager->MaxPool(n, inLayer->m_H, inLayer->m_W, inLayer->m_C, inLayer->m_Stride,
				inLayer->m_Size, inLayer->m_Pad, netState->m_InputGpu, inLayer->m_Output_Gpu);// , inLayer->m_Indexes_Gpu);

			break;

		default:
			break;
	}

	//const auto elapsed_time = std::chrono::steady_clock::now() - start_time;

	//auto timing = std::chrono::duration<double, std::milli>(elapsed_time).count();
	//printf("*******Layer{%d} exec time is %f : Diff : %f\n\n\n", netState->m_LayerIndex, timing, timing - timeAccumulator);

	return timeAccumulator;
}


//sprintf(debugFileName, "debug_im2ol_layer_%d.log", netState->m_LayerIndex);
//PrintOCLBuffer(netState->m_Workspace, m_OCLManager, debugFileName, inLayer->m_Inputs);


//sprintf(debugFileName, "debug_gemm_layer_%d.log", netState->m_LayerIndex);
//PrintOCLBuffer(inLayer->m_OutputSwapGPUBuffers[netState->m_ConvSwapBufIdx], m_OCLManager, debugFileName, inLayer->m_Outputs);

//sprintf(debugFileName, "debug_addbias_layer_%d.log", netState->m_LayerIndex);
//PrintOCLBuffer(inLayer->m_OutputSwapGPUBuffers[netState->m_ConvSwapBufIdx], m_OCLManager, debugFileName, inLayer->m_Outputs);

//sprintf(debugFileName, "debug_activate_layer_%d.log", netState->m_LayerIndex);
//PrintOCLBuffer(inLayer->m_OutputSwapGPUBuffers[netState->m_ConvSwapBufIdx], m_OCLManager, debugFileName, inLayer->m_Outputs);


//sprintf(debugFileName, "debug_activate_layer_%d.log", netState->m_LayerIndex);
//PrintOCLBuffer(inLayer->m_OutputSwapGPUBuffers[netState->m_ConvSwapBufIdx], m_OCLManager, debugFileName, inLayer->m_Inputs);
