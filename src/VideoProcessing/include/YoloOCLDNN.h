
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

#pragma once

#ifndef YOLO_NEURAL_NETWORK_H
#define YOLO_NEURAL_NETWORK_H

#include <vector>
#include <istream>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <sstream>
#include <omp.h> 
#include <numeric>
#include <cairo.h>
#include <mutex>
#include "SimpleIni.h"
#include "OCLManager.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/opencv.hpp"

#ifdef __linux__
#include <pthread.h>
#endif

extern "C" {

#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"
#include "libavdevice/avdevice.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"

}


#define DOABS					1
#define ENABLE_DISPLAY			(PROFILE_KERNELS == 1 || 0)

#define DST_WIDTH               416
#define DST_HEIGHT              416

#define MAX_FRAME_QUEUE_ITEMS   5



void EnumerateFilesInDirectory(string srcFolder,  vector<string> &fileNames, vector<string> &imageNames);

typedef enum {

	THREAD_STATUS_INITIALIZED	= 0,
	THREAD_STATUS_RUNNING		= 1,
	THREAD_STATUS_TERMINATED	= 2
}EnumThreadStatus;

typedef enum {

	LOG_MSG_TYPE_INFO		= 0,
	LOG_MSG_TYPE_ERROR		= 1,
	LOG_MSG_TYPE_WARNING	= 2,
	LOG_MSG_TYPE_DEBUG		= 3
}EnumLogMsgType;

typedef void(*LOGGERCALLBACK)(std::string logMsg, EnumLogMsgType logType);

typedef enum {

	YOLO_DNN_LAYER_CONVOLUTIONAL	= 0,
	YOLO_DNN_LAYER_MAXPOOL			= 1,
	YOLO_DNN_LAYER_REGION			= 2
} EnumYOLODeepNNLayerType;

typedef enum {

	YOLO_DNN_COST_TYPE_SSE		= 0, 
	YOLO_DNN_COST_TYPE_MASKED	= 1, 
	YOLO_DNN_COST_TYPE_SMOOTH	= 2
} EnumYOLODeepNNCostType;

typedef enum {

	YOLO_DNN_ACTIVATION_TYPE_LOGISTIC	= 0, 
	YOLO_DNN_ACTIVATION_TYPE_RELU		= 1, 
	YOLO_DNN_ACTIVATION_TYPE_RELIE		= 2, 
	YOLO_DNN_ACTIVATION_TYPE_LINEAR		= 3, 
	YOLO_DNN_ACTIVATION_TYPE_RAMP		= 4, 
	YOLO_DNN_ACTIVATION_TYPE_TANH		= 5, 
	YOLO_DNN_ACTIVATION_TYPE_PLSE		= 6, 
	YOLO_DNN_ACTIVATION_TYPE_LEAKY		= 7, 
	YOLO_DNN_ACTIVATION_TYPE_ELU		= 8, 
	YOLO_DNN_ACTIVATION_TYPE_LOGGY		= 9, 
	YOLO_DNN_ACTIVATION_TYPE_STAIR		= 10, 
	YOLO_DNN_ACTIVATION_TYPE_HARDTAN	= 11, 
	YOLO_DNN_ACTIVATION_TYPE_LHTAN		= 12
}EnumYOLODeepNNActivationType;

typedef struct {
	int		*m_Leaf;
	int		m_TotalN;
	int		*m_Parent;
	int		*m_Group;
	char	**m_Name;
	int		m_Groups;
	int		*m_GroupSize;
	int		*m_GroupOffset;
} StructHeirachicalTree;

typedef struct  {

	EnumYOLODeepNNLayerType			m_LayerType;
	EnumYOLODeepNNActivationType	m_Activation;
	int								m_BatchNormalize;
	int								m_Batch;
	int								m_Flipped;
	int								m_Inputs;
	int								m_Outputs;
	int								m_H;
	int								m_W;
	int								m_C;
	int								m_OutH;
	int								m_OutW;
	int								m_OutC;
	int								m_N;
	int								m_Size;
	int								m_Stride;
	int								m_Pad;
	int								m_Classes;
	int								m_Coords;
	int								m_ClassFix;
	int								m_DontLoad;
	int								m_DontLoadScales;
	float							*m_Biases;
	float							*m_Scales;
	float							*m_Weights;
	float							*m_Output;
	float							*m_RollingMean;
	float							*m_RollingVariance;
	size_t							m_Workspace_Size;
	OCLBuffer						*m_Weights_Gpu;
	OCLBuffer						*m_Biases_Gpu;
	OCLBuffer						*m_Biases_Sparse_Gpu;
	OCLBuffer						*m_Output_Gpu;
	OCLBuffer						*m_OutputSwapGPUBuffers[2];
	StructPinnedOCLBuffer			*m_PinnedBuffer;
	float							*m_PinnedOutput;
	

}StructYOLODeepNNLayer;


typedef struct {

	OCLBuffer				*m_Workspace;
	int						m_TotalLayers;
	int						m_BatchSize;
	StructYOLODeepNNLayer	*m_Layers;
	int						m_Outputs;
	int						m_TimeSteps;
	int						m_Step;
	int						m_Inputs;
	int						m_H;
	int                     m_W;
	int                     m_C;
	int						m_GpuIndex;
} StructYOLODeepNN;


typedef struct  {

	OCLBuffer			*m_InputGpu;
	OCLBuffer			*m_InputRefGpu;
	OCLBuffer			*m_Workspace;
	int					m_LayerIndex;
	int                 m_ConvSwapBufIdx;
	int                 m_InputSize;
	StructYOLODeepNN	*m_DeepNN;
}StructYOLODeepNNState;


typedef struct {
	int					m_Batch;
	int					m_Inputs;
	int					m_H;
	int					m_W;
	int					m_C;
	int					m_Index;
} StructLayerFeedParams;

typedef struct {
	
	float m_X;
	float m_Y;
	float m_W;
	float m_H;
}StructDetectionBBox;

typedef struct {

	int m_Index;
	int m_ClassIdx;
	float **m_ProbScores;
} StructSortableBBox;

typedef struct {

	int m_H;
	int m_W;
	int m_C;
	float *m_DataArray;
} StructImage;


typedef struct {

	StructImage		*m_SrcImage;
	StructImage		*m_TempImage;
	StructImage		*m_ResizedImage;
	cv::Mat			m_CurrentImageMat;
	cv::Mat			m_OverlayMat;
	cv::Mat			m_OverlayFinalMat;
	cv::Mat			m_DisplayImageMat;
	cv::Rect		m_OverlayRect;
	char			m_WorkingImageName[FILENAME_MAX];
	bool			m_SingletonSrcObject;
}StructRAWFrameSrcObject;

typedef struct {

	StructRAWFrameSrcObject *m_RAWSrcObject;
	StructYOLODeepNNLayer	*m_FinalLayer;
	StructDetectionBBox		*m_DetBBoxes;
	float					**m_DetProbScores;
	float					m_InferenceDuration;
}StructRAWFrameSinkObject;


class YOLONeuralNet {

private:

	std::vector<std::string>	m_ClassLabels;
	char						m_ClassLabelsFile[FILENAME_MAX];
	char						m_NetworkConfigFile[FILENAME_MAX];
	char						m_WeightsFile[FILENAME_MAX];

	std::vector<std::string>	m_LayerNames;
	CSimpleIniA					*m_IniReader;

	StructYOLODeepNN			*m_YOLODeepNN;
	OCLManager					*m_OCLManager;
	char						m_OCLDeviceName[128];
	cairo_surface_t				*m_CairoSurface;
	cairo_t						*m_Cairo;
	cv::Mat						m_CairoTarget;
	bool						m_EnableDisplay;
	bool                        m_SaveOutput;

	char						m_SrcVideoPath[FILENAME_MAX];
	char                        m_CurrImageName[FILENAME_MAX];
	char                        m_WorkingImageName[FILENAME_MAX];


	vector<string>				m_ImageBatch;
	vector<string>				m_ImageNames;

	bool						m_VideoFileEOS;
	AVFormatContext				*m_AVFormatContext;
	int							m_VideoStreamIdx;
	AVCodecContext				*m_AVCodecCtx;
	AVCodec						*m_AVCodec;
	AVFrame						*m_AVFrame;
	AVFrame						*m_AVFrameRGB;
	int							m_NumRGBbytes;
	uint8_t						*m_AVRGBBuffer;
	struct SwsContext			*m_ImgConvertCtx;
	float						m_DetThreshold;
	float						m_NMSOverlap;
	StructYOLODeepNNState		*m_YoloNNCurrentState;
	char						m_OutFolder[FILENAME_MAX];
	char						m_OverlayDeviceProp[256];
	std::queue<StructRAWFrameSrcObject*>	m_SrcFrameQueue;
	std::mutex					m_SrcFrameQueueMutex;
	std::queue<StructRAWFrameSinkObject*>	m_SinkFrameQueue;
	std::mutex					m_SinkFrameQueueMutex;
	bool						m_SinkActive;


	LOGGERCALLBACK				logWriteFunc;
	char						m_LogMsgStr[512];

	int                         m_SinkFrameCount;
	AVCodec						*m_AVSinkCodec;
	AVCodecContext				*m_AVSinkCodecContext;
	AVOutputFormat				*m_AVSinkFormat;
	AVFormatContext				*m_AVSinkFormatContext;
	AVStream					*m_AVSinkStream;
	AVFrame                     *m_AVSinkFrame;
	AVFrame                     *m_AVSinkRGBFrame;
	struct SwsContext			*m_SinkConvertCtx;
	int							m_SinkCopyYUVBytes;
	uint8_t						*m_SinkYUVBuffer;
	int							m_SinkCopyRGBBytes;
	uint8_t						*m_SinkRGBBuffer;
	bool						m_AVIHeaderWritten;

	int							m_FpsNum;
	int							m_FpsDen;

	EnumThreadStatus			m_SinkThreadStatus;

#ifdef __linux__
	pthread_t 			m_ProcSrcThread;
	pthread_t 			m_ProcSinkThread;
#endif



	static inline float LogisticActivate(float x) { return (float)(1. / (1. + exp(-x))); }

	bool ParseNetworkConfiguration();
	bool PrepareConvolutionalTypeLayer(int sectionIdx, int layerIdx, StructLayerFeedParams *layerFeedParams);
	bool PrepareRegionTypeLayer(int sectionIdx, int layerIdx, StructLayerFeedParams *layerFeedParams);
	bool PrepareMaxpoolTypeLayer(int sectionIdx, int layerIdx, StructLayerFeedParams *layerFeedParams);
	bool ParseNNLayers();
	bool ParseNNWeights();
	float PropagateLayerInputsForward(StructYOLODeepNNLayer *inLayer, StructYOLODeepNNState *netState);
	void GetDetectionBBoxes(StructYOLODeepNNLayer *nnLayer, int w, int h, float thresh, float **probs, StructDetectionBBox *bBoxes, int onlyObjectness, int *map);
	StructDetectionBBox GetRegionBBox(float *x, float *biases, int n, int index, int i, int j, int w, int h);
	void ApplyNMS(StructDetectionBBox *boxes, float **probs, int total, int classes, float thresh);
	void PutCairoOverlay(StructRAWFrameSrcObject *srcRAWFrameObject, std::string const& timeText, cv::Point2d timeCenterPoint, std::string const& fontFace, double fontSize,
		cv::Scalar textColor, bool fontItalic, bool fontBold);
	void RunInference(StructRAWFrameSrcObject *rawFrameObject, float &inferenceDuration, 
		float **detProbScores, StructDetectionBBox *detBBoxes);
	
	

public:

	YOLONeuralNet(LOGGERCALLBACK loggerCallback, char* classLabelsFile, char *networkConfigFile, char *weightsFile,
		bool display, bool saveOutput, float threshold, float nmsOverlap);
	~YOLONeuralNet();

	inline int GetDNNWidth() { return m_YOLODeepNN->m_W; };
	inline int GetDNNHeight() { return m_YOLODeepNN->m_H; };
	inline bool SinkAlive() { return m_SinkActive; };
	inline int GetFPSNum() { return m_FpsNum; };
	inline int GetFPSDen() { return m_FpsDen; };

	bool Initialize();
	void Finalize();
	void ProcessSingleImage(char* inputFile);
	void ProcessImageBatch(char *srcFolder);
	void ProcessVideo(char *srcVideoPath);
	void PostProcessDetections(StructRAWFrameSinkObject *rawFrameSinkObject);
	bool OpenVideoFile(cv::Mat &dstMat);
	bool FetchNextFrameFromVideo(cv::Mat &dstMat);
	void CloseVideoFile();
	int GetRemainingImagesCount();
	void FetchNextImage(char *outImagePath, char *outImageName);
	void ProcessSinkFramesInSequence();
	void CopyVideoFileName(char *dstFilePath);
	void SignalEOS();
	void EnqueueRAWFrame(StructRAWFrameSrcObject *rawFrameObject);
	bool InitializeSinkResources(StructRAWFrameSrcObject *rawFrameObject, int fpsNum, int fpsDen);
	bool ProcessSinkFrame(StructRAWFrameSinkObject *rawSinkFrameObject);
	void FinalizeSinkResources();
	void WaitForSync();


};


#endif
