// Original arboOCR code (not a RapidOcrOnnx port) — see preprocess.hpp.
#include "arboOCR/preprocess.hpp"

#include <opencv2/imgproc.hpp>

namespace arbo::ocr {

cv::Mat applyClahe(const cv::Mat& src) {
    if (src.empty()) return src;
    if (src.channels() != 1 && src.channels() != 3) return src;

    auto clahe = cv::createCLAHE(/*clipLimit=*/2.0, cv::Size(8, 8));

    if (src.channels() == 1) {
        cv::Mat out;
        clahe->apply(src, out);
        return out;
    }

    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    clahe->apply(channels[0], channels[0]);
    cv::merge(channels, lab);

    cv::Mat out;
    cv::cvtColor(lab, out, cv::COLOR_Lab2BGR);
    return out;
}

} // namespace arbo::ocr
