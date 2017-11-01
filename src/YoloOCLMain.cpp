
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


#include <stdio.h>
#include "YoloOCLDNN.h"


#ifdef _DEBUG
//#include <vld.h>
#endif


YOLONeuralNet	*m_YOLODeepNNObj;
char			m_LogFilePath[FILENAME_MAX];
fstream			m_LogFile;
std::mutex		m_LogFileMutex;

std::string GetCurrentDateTime() {

	time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(&now);
	strftime(buf, sizeof(buf), "%Y-%m-%d %X ", &tstruct);
	return buf;
}


void logWrite(std::string logMsg, EnumLogMsgType logType=EnumLogMsgType::LOG_MSG_TYPE_INFO) {

	m_LogFileMutex.lock();
	m_LogFile.open(m_LogFilePath, ios_base::out | ios_base::in | ios_base::app);
	bool logFileExists = m_LogFile.is_open();

	if(!logFileExists)
		m_LogFile.open(m_LogFilePath, ios_base::out); 

	std::string formattedStr = "";
	
	switch (logType) {

		case EnumLogMsgType::LOG_MSG_TYPE_INFO:
			formattedStr = GetCurrentDateTime() + "[INFO] " + logMsg + "\n";
			break;

		case EnumLogMsgType::LOG_MSG_TYPE_ERROR:
			formattedStr = GetCurrentDateTime() + "[ERROR] " + logMsg + "\n";
			break;

		case EnumLogMsgType::LOG_MSG_TYPE_WARNING:
			formattedStr = GetCurrentDateTime() + "[WARNING] " + logMsg + "\n";
			break;

		case EnumLogMsgType::LOG_MSG_TYPE_DEBUG:
			formattedStr = GetCurrentDateTime() + "[DEBUG] " + logMsg + "\n";
			break;
	}

	
	m_LogFile.write(formattedStr.c_str(), formattedStr.length());
	m_LogFile.close();
	printf("%s", formattedStr.c_str());
	m_LogFileMutex.unlock();
}

inline bool FileExists(const std::string& name) {

	ifstream f(name.c_str());
	return f.good();
}


int main(int argc, char* argv[]) {

	string	currentDir = ExePath();
	char	labelsFile[FILENAME_MAX];
	char	configFile[FILENAME_MAX];
	char	weightsFile[FILENAME_MAX];
	char	inputImage[FILENAME_MAX];
	char	inputFolder[FILENAME_MAX];
	char	inputVideo[FILENAME_MAX];
	char	logMsg[512];
	char    gpuDevType[64];
	int		enableDisplay = 0;
	int		saveOutput = 0;
	int     inputType = 0;
	float   detThreshold = 0.2f;
	float   nmsOverlap = 0.45f;
	EnumSaveOPType saveOPType = EnumSaveOPType::SAVE_OUTPUT_TYPE_NONE;

	sprintf(gpuDevType, "AMD");

#ifdef WIN32
	sprintf(m_LogFilePath, "%s\\YoloOCLInference.log", currentDir.c_str());
#elif __linux__
	strcpy(m_LogFilePath, "YoloOCLInference.log");
#endif

	std::remove(m_LogFilePath);
	logWrite("*************START*************");

	for (int i = 1; i < argc; i++) {

		if (strcmp(argv[i], "-input") == 0) {

			if (++i >= argc) {

				sprintf(logMsg, "ERROR - Invalid param for %s", argv[i - 1]);
				logWrite(std::string(logMsg), EnumLogMsgType::LOG_MSG_TYPE_ERROR);
				return -1;
			}
			strcpy(inputImage, argv[i]);
			inputType |= 1 << 0;
		}
		else if (strcmp(argv[i], "-folder") == 0) {

			if (++i >= argc) {

				sprintf(logMsg, "ERROR - Invalid param for %s", argv[i - 1]);
				logWrite(std::string(logMsg), EnumLogMsgType::LOG_MSG_TYPE_ERROR);
				return -1;
			}
			strcpy(inputFolder, argv[i]);
			inputType |= 1 << 1;
		}
		else if (strcmp(argv[i], "-video") == 0) {

			if (++i >= argc) {

				sprintf(logMsg, "ERROR - Invalid param for %s", argv[i - 1]);
				logWrite(std::string(logMsg), EnumLogMsgType::LOG_MSG_TYPE_ERROR);
				return -1;
			}
			strcpy(inputVideo, argv[i]);
			inputType |= 1 << 2;
		}
		else if (strcmp(argv[i], "-display") == 0) {

			if (++i >= argc || sscanf(argv[i], "%d", &enableDisplay) != 1) {

				sprintf(logMsg, "ERROR - Invalid param for %s", argv[i - 1]);
				logWrite(std::string(logMsg), EnumLogMsgType::LOG_MSG_TYPE_ERROR);
				return -1;
			}
		}
		else if (strcmp(argv[i], "-save") == 0) {

			if (++i >= argc || sscanf(argv[i], "%d", &saveOutput) != 1) {

				sprintf(logMsg, "ERROR - Invalid param for %s", argv[i - 1]);
				logWrite(std::string(logMsg), EnumLogMsgType::LOG_MSG_TYPE_ERROR);
				return -1;
			}

			if (saveOutput == 0) {
				
				saveOPType = EnumSaveOPType::SAVE_OUTPUT_TYPE_NONE;
				logWrite("Saving output disabled", EnumLogMsgType::LOG_MSG_TYPE_ERROR);
			}
			else if (saveOutput == 1) {
			
				saveOPType = EnumSaveOPType::SAVE_OUTPUT_TYPE_JPEG;
				logWrite("Saving JPEG images as output", EnumLogMsgType::LOG_MSG_TYPE_ERROR);
			}
			else if (saveOutput == 2) {
			
				saveOPType = EnumSaveOPType::SAVE_OUTPUT_TYPE_VIDEO;
				logWrite("Saving video file as output", EnumLogMsgType::LOG_MSG_TYPE_ERROR);
			}

		}
		else if (strcmp(argv[i], "-det_threshold") == 0) {

			if (++i >= argc || sscanf(argv[i], "%f", &detThreshold) != 1) {

				sprintf(logMsg, "ERROR - Invalid param for %s", argv[i - 1]);
				logWrite(std::string(logMsg), EnumLogMsgType::LOG_MSG_TYPE_ERROR);
				return -1;
			}
		}
		else if (strcmp(argv[i], "-nms_overlap") == 0) {

			if (++i >= argc || sscanf(argv[i], "%f", &nmsOverlap) != 1) {

				sprintf(logMsg, "ERROR - Invalid param for %s", argv[i - 1]);
				logWrite(std::string(logMsg), EnumLogMsgType::LOG_MSG_TYPE_ERROR);
				return -1;
			}
		}
		else if (strcmp(argv[i], "-gpu_type") == 0) {

			if (++i >= argc) {

				sprintf(logMsg, "ERROR - Invalid param for %s", argv[i - 1]);
				logWrite(std::string(logMsg), EnumLogMsgType::LOG_MSG_TYPE_ERROR);
				return -1;
			}
			strcpy(gpuDevType, argv[i]);
		}
	}

	if ((((inputType >> 0) & 1) && !FileExists(inputImage))
		|| (((inputType >> 2) & 1) && !FileExists(inputVideo))) {

		sprintf(logMsg, "ERROR - File doesnot exist %s", ((inputType >> 0) & 1) ? inputImage : inputVideo);
		logWrite(std::string(logMsg), EnumLogMsgType::LOG_MSG_TYPE_ERROR);
		return -1;
	}

#ifdef WIN32

	sprintf(labelsFile, "%s\\coco.names", currentDir.c_str());
	sprintf(configFile, "%s\\tiny-yolo.cfg", currentDir.c_str());
	sprintf(weightsFile, "%s\\tiny-yolo.weights", currentDir.c_str());
#elif __linux__

	strcpy(labelsFile, "coco.names");
	strcpy(configFile, "tiny-yolo.cfg");
	strcpy(weightsFile, "tiny-yolo.weights");
#endif
	

	logWrite("Creating YOLONeuralNet Object", EnumLogMsgType::LOG_MSG_TYPE_INFO);
	m_YOLODeepNNObj = new YOLONeuralNet(logWrite, gpuDevType, labelsFile, configFile, weightsFile,
		(enableDisplay == 1)?true:false, saveOPType, detThreshold, nmsOverlap);
	if (m_YOLODeepNNObj->Initialize()) {

		if ((inputType >> 0) & 1) {

			logWrite("YOLONeuralNet - Operating in single image mode", EnumLogMsgType::LOG_MSG_TYPE_INFO);
			m_YOLODeepNNObj->ProcessSingleImage(inputImage);
		}
		else if ((inputType >> 1) & 1) {

			logWrite("YOLONeuralNet - Operating in batch image mode", EnumLogMsgType::LOG_MSG_TYPE_INFO);
			m_YOLODeepNNObj->ProcessImageBatch(inputFolder);
		}
		else if ((inputType >> 2) & 1) {

			logWrite("YOLONeuralNet - Operating in video file mode", EnumLogMsgType::LOG_MSG_TYPE_INFO);
			m_YOLODeepNNObj->ProcessVideo(inputVideo);
		}
		m_YOLODeepNNObj->Finalize();
	}
	
	delete m_YOLODeepNNObj;

	logWrite("*************END*************");

	return 0;
}

