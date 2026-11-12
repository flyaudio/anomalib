/**
 * PatchCore TensorRT 10 inference example (C++).
 *
 * Loads a TensorRT engine built from an anomalib-exported PatchCore ONNX model
 * and runs single-image inference:
 *
 *   preprocessing (OpenCV):  BGR -> RGB -> resize 256x256 (INTER_AREA, approx.
 *                            torchvision antialias bilinear) -> center crop
 *                            224x224 -> ImageNet normalization -> CHW float32
 *   TensorRT:                feature extraction + memory-bank nearest-neighbor
 *                            search + anomaly map + post-processing, all in-graph
 *   post-processing:         normalized score / label / anomaly map (all outputs
 *                            are already normalized to [0,1] by the in-graph
 *                            anomalib PostProcessor, threshold == 0.5)
 *
 * Expected ONNX signature (see export_patchcore_onnx.py):
 *   input :  "input"        float32  (N, 3, 224, 224)   (batch may be dynamic)
 *   outputs: "pred_score"   float32  (N,)
 *            "anomaly_map"  float32  (N, 1, 224, 224)
 *            "pred_label"   bool     (N,)
 *            "pred_mask"    bool     (N, 1, 224, 224)
 *
 * Tested API level: TensorRT 10.11 (enqueueV3 / tensor-name based IO).
 *
 * Build: see CMakeLists.txt / README.md.
 */

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// Configuration (must match the export settings in export_patchcore_onnx.py)
// ---------------------------------------------------------------------------
constexpr int kResizeTo = 256;   // resize before crop
constexpr int kInputH = 224;     // network input height
constexpr int kInputW = 224;     // network input width
constexpr float kImageNetMean[3] = {0.485f, 0.456f, 0.406f};  // R, G, B
constexpr float kImageNetStd[3] = {0.229f, 0.224f, 0.225f};

// ---------------------------------------------------------------------------
// TensorRT logger
// ---------------------------------------------------------------------------
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

struct TrtDeleter {
    template <typename T>
    void operator()(T* obj) const { delete obj; }
};
template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        const cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                                    \
            std::cerr << "CUDA error: " << cudaGetErrorString(err_) << " at "         \
                      << __FILE__ << ":" << __LINE__ << std::endl;                    \
            std::exit(EXIT_FAILURE);                                                  \
        }                                                                             \
    } while (0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
size_t elementSize(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kBOOL: return 1;
        case nvinfer1::DataType::kUINT8: return 1;
        default: throw std::runtime_error("Unsupported tensor data type");
    }
}

int64_t volume(const nvinfer1::Dims& dims) {
    int64_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) v *= dims.d[i];
    return v;
}

std::vector<char> readBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open engine file: " + path);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<size_t>(size));
    if (!file.read(buffer.data(), size)) throw std::runtime_error("Failed to read: " + path);
    return buffer;
}

// ---------------------------------------------------------------------------
// Preprocessing: BGR uint8 image -> CHW float32 normalized tensor
// ---------------------------------------------------------------------------
std::vector<float> preprocess(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // INTER_AREA is the closest OpenCV match to torchvision's bilinear +
    // antialias=True when downscaling.
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(kResizeTo, kResizeTo), 0, 0, cv::INTER_AREA);

    const int offset = (kResizeTo - kInputH) / 2;
    cv::Mat crop = resized(cv::Rect(offset, offset, kInputW, kInputH));

    cv::Mat floatImg;
    crop.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(floatImg, channels);  // R, G, B

    std::vector<float> out(3 * kInputH * kInputW);
    const size_t planeSize = static_cast<size_t>(kInputH) * kInputW;
    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - kImageNetMean[c]) / kImageNetStd[c];
        std::memcpy(out.data() + c * planeSize, channels[c].data, planeSize * sizeof(float));
    }
    return out;
}

// ---------------------------------------------------------------------------
// A TensorRT tensor binding (device buffer + staging host buffer)
// ---------------------------------------------------------------------------
struct Binding {
    std::string name;
    bool isInput = false;
    nvinfer1::DataType dtype{};
    nvinfer1::Dims dims{};
    size_t bytes = 0;
    void* device = nullptr;
    std::vector<uint8_t> host;
};

class PatchCoreEngine {
public:
    explicit PatchCoreEngine(const std::string& enginePath) {
        const std::vector<char> blob = readBinaryFile(enginePath);
        runtime_.reset(createInferRuntime(gLogger));
        if (!runtime_) throw std::runtime_error("Failed to create TensorRT runtime");
        engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
        if (!engine_) throw std::runtime_error("Failed to deserialize engine (built for a different GPU/TRT version?)");
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("Failed to create execution context");
        CUDA_CHECK(cudaStreamCreate(&stream_));
        setupBindings();
    }

    ~PatchCoreEngine() {
        for (auto& b : bindings_) if (b.device) cudaFree(b.device);
        if (stream_) cudaStreamDestroy(stream_);
    }

    // Runs inference on one preprocessed image, returns elapsed GPU time (ms).
    float infer(const std::vector<float>& inputData) {
        Binding& input = bindings_[inputIndex_];
        if (inputData.size() * sizeof(float) != input.bytes) {
            throw std::runtime_error("Input size mismatch: engine expects " +
                                     std::to_string(input.bytes) + " bytes");
        }
        CUDA_CHECK(cudaMemcpyAsync(input.device, inputData.data(), input.bytes,
                                   cudaMemcpyHostToDevice, stream_));

        const auto t0 = std::chrono::steady_clock::now();
        if (!context_->enqueueV3(stream_)) throw std::runtime_error("enqueueV3 failed");
        for (auto& b : bindings_) {
            if (!b.isInput) {
                CUDA_CHECK(cudaMemcpyAsync(b.host.data(), b.device, b.bytes,
                                           cudaMemcpyDeviceToHost, stream_));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<float, std::milli>(t1 - t0).count();
    }

    const Binding* findOutput(const std::string& namePart) const {
        for (const auto& b : bindings_) {
            if (!b.isInput && b.name.find(namePart) != std::string::npos) return &b;
        }
        return nullptr;
    }

private:
    void setupBindings() {
        const int nb = engine_->getNbIOTensors();

        // Pass 1: resolve dynamic input dims (batch -> 1, H/W -> 224) so that
        // output shapes can be queried in pass 2.
        for (int i = 0; i < nb; ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) continue;
            nvinfer1::Dims shape = engine_->getTensorShape(name);
            for (int d = 0; d < shape.nbDims; ++d) {
                if (shape.d[d] < 0) {
                    shape.d[d] = (d == 0) ? 1 : (d == 2 ? kInputH : (d == 3 ? kInputW : 1));
                }
            }
            if (!context_->setInputShape(name, shape)) {
                throw std::runtime_error(std::string("setInputShape failed for ") + name);
            }
        }

        // Pass 2: allocate device buffers with the now-resolved shapes.
        for (int i = 0; i < nb; ++i) {
            const char* rawName = engine_->getIOTensorName(i);
            Binding b;
            b.name = rawName;
            b.isInput = engine_->getTensorIOMode(rawName) == nvinfer1::TensorIOMode::kINPUT;
            b.dtype = engine_->getTensorDataType(rawName);
            b.dims = context_->getTensorShape(rawName);
            b.bytes = static_cast<size_t>(volume(b.dims)) * elementSize(b.dtype);
            CUDA_CHECK(cudaMalloc(&b.device, b.bytes));
            if (!b.isInput) b.host.resize(b.bytes);

            std::cout << "[Binding] " << (b.isInput ? "input " : "output") << " '" << b.name
                      << "' shape=(";
            for (int d = 0; d < b.dims.nbDims; ++d) std::cout << b.dims.d[d] << (d + 1 < b.dims.nbDims ? "," : "");
            std::cout << ") " << b.bytes << " bytes" << std::endl;

            if (b.isInput) inputIndex_ = static_cast<int>(bindings_.size());
            bindings_.push_back(std::move(b));
        }

        // Pass 3: bind addresses.
        for (auto& b : bindings_) {
            if (!context_->setTensorAddress(b.name.c_str(), b.device)) {
                throw std::runtime_error("setTensorAddress failed for " + b.name);
            }
        }
    }

    TrtUniquePtr<nvinfer1::IRuntime> runtime_;
    TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
    TrtUniquePtr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_ = nullptr;
    std::vector<Binding> bindings_;
    int inputIndex_ = -1;
};

// ---------------------------------------------------------------------------
// Post-processing & visualization
// ---------------------------------------------------------------------------
void visualize(const PatchCoreEngine& eng, const cv::Mat& original,
               const std::string& outPath, float threshold) {
    const Binding* score = eng.findOutput("pred_score");
    const Binding* label = eng.findOutput("pred_label");
    const Binding* amap = eng.findOutput("anomaly_map");
    const Binding* pmask = eng.findOutput("pred_mask");

    const float predScore = score ? reinterpret_cast<const float*>(score->host.data())[0] : -1.0f;
    const bool anomalous = label ? (label->host[0] != 0) : (predScore > threshold);

    std::cout << "  pred_score = " << predScore
              << "  (threshold " << threshold << ")  ->  "
              << (anomalous ? "ANOMALOUS" : "normal") << std::endl;

    cv::Mat vis = original.clone();

    if (amap) {
        // anomaly_map is normalized to [0,1] by the in-graph PostProcessor.
        const int h = amap->dims.nbDims >= 4 ? static_cast<int>(amap->dims.d[2]) : kInputH;
        const int w = amap->dims.nbDims >= 4 ? static_cast<int>(amap->dims.d[3]) : kInputW;
        cv::Mat mapF32(h, w, CV_32FC1);
        std::memcpy(mapF32.data, amap->host.data(), static_cast<size_t>(h) * w * sizeof(float));
        cv::Mat map8u;
        mapF32.convertTo(map8u, CV_8UC1, 255.0);  // already [0,1]; convertTo clamps via saturate_cast
        cv::resize(map8u, map8u, original.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat heatmap;
        cv::applyColorMap(map8u, heatmap, cv::COLORMAP_JET);
        cv::addWeighted(original, 0.5, heatmap, 0.5, 0.0, vis);
    }

    if (pmask) {
        const int h = pmask->dims.nbDims >= 4 ? static_cast<int>(pmask->dims.d[2]) : kInputH;
        const int w = pmask->dims.nbDims >= 4 ? static_cast<int>(pmask->dims.d[3]) : kInputW;
        cv::Mat mask(h, w, CV_8UC1);
        for (int i = 0; i < h * w; ++i) mask.data[i] = pmask->host[i] ? 255 : 0;
        cv::resize(mask, mask, original.size(), 0, 0, cv::INTER_NEAREST);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(vis, contours, -1, cv::Scalar(0, 0, 255), 2);  // red = anomaly
    }

    cv::putText(vis, (anomalous ? "ANOMALOUS " : "normal ") + std::to_string(predScore).substr(0, 5),
                {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                anomalous ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 180, 0), 2);
    cv::imwrite(outPath, vis);
    std::cout << "  saved: " << outPath << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage:\n  patchcore_trt <engine> <image_or_dir> [--out out_dir] "
                     "[--threshold 0.5]\n"
                     "\nExample:\n  patchcore_trt patchcore_fp16.engine ./test/broken_large --out ./results"
                  << std::endl;
        return EXIT_FAILURE;
    }
    const std::string enginePath = argv[1];
    const std::string imagePath = argv[2];
    std::string outDir = "./patchcore_results";
    float threshold = 0.5f;  // anomalib PostProcessor maps the adaptive threshold to 0.5
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) outDir = argv[++i];
        if (a == "--threshold" && i + 1 < argc) threshold = std::stof(argv[++i]);
    }
    fs::create_directories(outDir);

    try {
        PatchCoreEngine engine(enginePath);

        // Collect images.
        std::vector<fs::path> images;
        if (fs::is_directory(imagePath)) {
            for (const auto& e : fs::directory_iterator(imagePath)) {
                const auto ext = e.path().extension().string();
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
                    images.push_back(e.path());
                }
            }
            std::sort(images.begin(), images.end());
        } else {
            images.push_back(imagePath);
        }
        if (images.empty()) throw std::runtime_error("No images found: " + imagePath);

        double totalMs = 0.0;
        int timed = 0;
        for (size_t i = 0; i < images.size(); ++i) {
            cv::Mat img = cv::imread(images[i].string(), cv::IMREAD_COLOR);
            if (img.empty()) { std::cerr << "  cannot read " << images[i] << std::endl; continue; }

            const std::vector<float> input = preprocess(img);
            const float ms = engine.infer(input);

            // First few runs are warmup (TensorRT autotuning caches, allocator, etc.)
            if (i >= 3) { totalMs += ms; ++timed; }
            std::cout << "[" << i + 1 << "/" << images.size() << "] " << images[i].filename().string()
                      << "  (" << ms << " ms)" << std::endl;

            const fs::path outPath = fs::path(outDir) / (images[i].stem().string() + "_vis.png");
            visualize(engine, img, outPath.string(), threshold);
        }
        if (timed > 0) {
            std::cout << "\nAverage latency (excluding " << std::min<size_t>(3, images.size())
                      << " warmup): " << totalMs / timed << " ms/image" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
