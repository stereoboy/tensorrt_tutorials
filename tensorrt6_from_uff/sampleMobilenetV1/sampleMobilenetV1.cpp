/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

//!
//! SampleUffSSD.cpp
//! This file contains the implementation of the Uff SSD sample. It creates the network using
//! the SSD UFF model.
//! It can be run with the following command line:
//! Command: ./sample_uff_ssd [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//!

#include "BatchStream.h"
#include "EntropyCalibrator.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvInfer.h"
#include "NvUffParser.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <fstream>
#include <sstream>

const std::string gSampleName = "TensorRT.sample_uff_ssd";

//!
//! \brief The SampleUffSSDParams structure groups the additional parameters required by
//!         the Uff SSD sample.
//!
struct SampleUffSSDParams : public samplesCommon::SampleParams
{
    std::string uffFileName;    //!< The file name of the UFF model to use
    std::string labelsFileName; //!< The file namefo the class labels
    int outputClsSize;          //!< The number of output classes
    int calBatchSize;           //!< The size of calibration batch
    int nbCalBatches;           //!< The number of batches for calibration
    int keepTopK;               //!< The maximum number of detection post-NMS
    float visualThreshold;      //!< The minimum score threshold to consider a detection
};

//! \brief  The SampleUffSSD class implements the SSD sample
//!
//! \details It creates the network using an UFF model
//!
class SampleUffSSD
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleUffSSD(const SampleUffSSDParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief Cleans up any state created in the sample class
    //!
    bool teardown();

private:
    SampleUffSSDParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    std::vector<samplesCommon::PPM<3, 300, 300>> mPPMs; //!< PPMs of test images

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an UFF model for SSD and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvuffparser::IUffParser>& parser);
    bool constructNetwork(void);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Filters output detections and verify results
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the SSD network by parsing the UFF model and builds
//!          the engine that will be used to run SSD (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleUffSSD::build()
{
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }
#if 0
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
#else
    auto constructed = constructNetwork();
#endif
    if (!constructed)
    {
        return false;
    }

#if 0
    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    assert(network->getNbOutputs() == 2);
#else

#endif
    return true;
}

//!
//! \brief Uses a UFF parser to create the SSD Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the SSD network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleUffSSD::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvuffparser::IUffParser>& parser)
{
    parser->registerInput(mParams.inputTensorNames[0].c_str(), DimsCHW(3, 300, 300), nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput(mParams.outputTensorNames[0].c_str());

    auto parsed = parser->parse(locateFile(mParams.uffFileName, mParams.dataDirs).c_str(), *network, DataType::kFLOAT);
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(1_GiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

    if (mParams.int8)
    {
        gLogInfo << "Using Entropy Calibrator 2" << std::endl;
        const std::string listFileName = "list.txt";
        const int imageC = 3;
        const int imageH = 300;
        const int imageW = 300;
        nvinfer1::DimsNCHW imageDims{};
        imageDims = nvinfer1::DimsNCHW{mParams.calBatchSize, imageC, imageH, imageW};
        BatchStream calibrationStream(mParams.nbCalBatches, imageDims, listFileName, mParams.dataDirs);
        calibrator.reset(
            new Int8EntropyCalibrator2<BatchStream>(calibrationStream, 0, "UffSSD", mParams.inputTensorNames[0].c_str()));
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
    }


    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        std::cerr<< "deserializeEngine Failed" << std::endl;
        return false;
    }

    return true;
}

bool SampleUffSSD::constructNetwork()
{
    void *dlh = dlopen("./build/plugins/NMSOptPlugin/libnmsoptplugin.so", RTLD_LAZY);
    if (nullptr == dlh)
    {
        std::cerr<< "Failed to load plugin libnmsoptplugin.so." << std::endl;
        return false;
    }

    IRuntime* runtime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());
    if (nullptr == runtime)
    {
        std::cerr<< "Failed to create InferRuntime." << std::endl;
        return false;
    }

    std::vector<char> trtEngineStream;
    std::string engineName = "./SSDMobileNet/ssd-small-SingleStream-gpu-b1-int8.plan";
    std::ifstream file(engineName, std::ios::binary);
    if (!file.good())
    {
        std::cerr<< "could not find serialized plan file." << std::endl;
        return false;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    trtEngineStream.resize(size);
    file.read(trtEngineStream.data(), size);
    file.close();

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trtEngineStream.data(), size, nullptr), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        std::cerr<< "deserializeEngine Failed" << std::endl;
        return false;
    }

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleUffSSD::infer()
{
    std::cout << ">>>" << __FUNCTION__ << std::endl;
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();
    std::cout<< "buffers.copyInputToDevice();" << std::endl;

    bool status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }
    std::cout<< "context->execute(...);" << std::endl;

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();
    std::cout<< "buffers.copyOutputToHost();" << std::endl;

    // Post-process detections and verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    std::cout << "<<<" << __FUNCTION__ << std::endl;
    return true;
}

//!
//! \brief Cleans up any state created in the sample class
//!
bool SampleUffSSD::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvuffparser::shutdownProtobufLibrary();
    return true;
}

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool SampleUffSSD::processInput(const samplesCommon::BufferManager& buffers)
{
#if 0
    const int inputC = mInputDims.d[0];
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];
    const int batchSize = mParams.batchSize;
#else
    std::cout << ">>>" << __FUNCTION__ << std::endl;
    const int inputC = 3;
    const int inputH = 300;
    const int inputW = 300;
    const int batchSize = mParams.batchSize;
#endif

    // Available images
    std::vector<std::string> imageList = {"dog.ppm", "bus.ppm"};
    mPPMs.resize(batchSize);
    assert(mPPMs.size() <= imageList.size());
    for (int i = 0; i < batchSize; ++i)
    {
        readPPMFile(locateFile(imageList[i], mParams.dataDirs), mPPMs[i]);
    }

#if 0
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    // Host memory for input buffer
    for (int i = 0, volImg = inputC * inputH * inputW; i < mParams.batchSize; ++i)
    {
        for (int c = 0; c < inputC; ++c)
        {
            // The color image to input should be in BGR order
            for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
            {
                hostDataBuffer[i * volImg + c * volChl + j]
                    = (2.0 / 255.0) * float(mPPMs[i].buffer[j * inputC + c]) - 1.0;
            }
        }
    }
#else
    int* hostDataBuffer = static_cast<int*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    // Host memory for input buffer
    for (int i = 0, volImg = inputC * inputH * inputW; i < mParams.batchSize; ++i)
    {
        for (int c = 0; c < inputC; ++c)
        {
            // The color image to input should be in BGR order
            for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
            {
                hostDataBuffer[i * volImg + c * volChl + j]
                    = int((2.0 / 255.0) * float(mPPMs[i].buffer[j * inputC + c]) - 1.0);
            }
        }
    }
#endif

    std::cout << "<<<" << __FUNCTION__ << std::endl;
    return true;
}

//!
//! \brief Filters output detections and verify result
//!
//! \return whether the detection output matches expectations
//!
bool SampleUffSSD::verifyOutput(const samplesCommon::BufferManager& buffers)
{
#if 0
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];
#else
    const int inputH = 300;
    const int inputW = 300;
#endif
    const int batchSize = mParams.batchSize;
    const int keepTopK = mParams.keepTopK;
    const float visualThreshold = mParams.visualThreshold;
    const int outputClsSize = mParams.outputClsSize;

    const float* detectionOut = static_cast<const float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    const int* keepCount = static_cast<const int*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));

    std::vector<std::string> classes(outputClsSize);

    // Gather class labels
    std::ifstream labelFile(locateFile(mParams.labelsFileName, mParams.dataDirs));
    string line;
    int id = 0;
    while (getline(labelFile, line))
    {
        classes[id++] = line;
    }

    bool pass = true;

    for (int p = 0; p < batchSize; ++p)
    {
        int numDetections = 0;
        // at least one correct detection
        bool correctDetection = false;

        for (int i = 0; i < keepCount[p]; ++i)
        {
            const float* det = &detectionOut[0] + (p * keepTopK + i) * 7;
            if (det[2] < visualThreshold)
            {
                continue;
            }

            // Output format for each detection is stored in the below order
            // [image_id, label, confidence, xmin, ymin, xmax, ymax]
            int detection = det[1];
            assert(detection < outputClsSize);
            std::string storeName = classes[detection] + "-" + std::to_string(det[2]) + ".ppm";

            numDetections++;
            if ((p == 0 && classes[detection] == "dog")
                || (p == 1 && (classes[detection] == "truck" || classes[detection] == "car")))
            {
                correctDetection = true;
            }

            gLogInfo << "Detected " << classes[detection].c_str() << " in the image " << int(det[0]) << " ("
                     << mPPMs[p].fileName.c_str() << ")"
                     << " with confidence " << det[2] * 100.f << " and coordinates (" << det[3] * inputW << ","
                     << det[4] * inputH << ")"
                     << ",(" << det[5] * inputW << "," << det[6] * inputH << ")." << std::endl;

            gLogInfo << "Result stored in " << storeName.c_str() << "." << std::endl;

            samplesCommon::writePPMFileWithBBox(
                storeName, mPPMs[p], {det[3] * inputW, det[4] * inputH, det[5] * inputW, det[6] * inputH});
        }
        pass &= correctDetection;
        pass &= numDetections >= 1;
    }

    return pass;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
SampleUffSSDParams initializeSampleParams(const samplesCommon::Args& args)
{
    SampleUffSSDParams params;
#if 0
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/ssd/");
        params.dataDirs.push_back("data/ssd/VOC2007/");
        params.dataDirs.push_back("data/ssd/VOC2007/PPMImages/");
        params.dataDirs.push_back("data/samples/ssd/");
        params.dataDirs.push_back("data/samples/ssd/VOC2007/");
        params.dataDirs.push_back("data/samples/ssd/VOC2007/PPMImages/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
#else
    params.dataDirs.push_back("./data/ssd/");
#endif
    params.uffFileName = "sample_ssd_relu6.uff";
    params.labelsFileName = "ssd_coco_labels.txt";
    params.inputTensorNames.push_back("Input");
#if 0
    params.batchSize = 2;
#else
    params.batchSize = 1;
#endif
    params.outputTensorNames.push_back("NMS");
    params.outputTensorNames.push_back("NMS_1");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    params.outputClsSize = 91;
    params.calBatchSize = 10;
    params.nbCalBatches = 10;
    params.keepTopK = 100;
    params.visualThreshold = 0.5;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_uff_ssd [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "data/samples/ssd/ and data/ssd/"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--fp16          Specify to run in fp16 mode." << std::endl;
    std::cout << "--int8          Specify to run in int8 mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    SampleUffSSD sample(initializeSampleParams(args));

    gLogInfo << "Building and running a GPU inference engine for SSD" << std::endl;

    if (!sample.build())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.teardown())
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
}