// Copyright (C) 2014 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#ifndef THEIA_IMAGE_IMAGE_H_
#define THEIA_IMAGE_IMAGE_H_

#include <cimg/CImg.h>
#include <Eigen/Core>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "theia/util/util.h"

namespace theia {
template <typename T> class Image;
typedef Image<float> FloatImage;
typedef Image<uchar> UcharImage;
typedef Image<int> IntImage;

// Templated on the number of channels.
template <typename T> class Image {
 public:
  Image();

  // Read from file.
  explicit Image(const std::string& filename);
  Image(const int width, const int height, const int channels);

  // Copy function. This is a deep copy of the image.
  Image(const Image<T>& image_to_copy);
  // Copy function. This is a deep copy of the image.
  template <typename D>
  explicit Image(const Image<D>& image_to_copy);
  explicit Image(const cimg_library::CImg<T>& image);
  Image<T>& operator=(const Image<T>& image2);
  ~Image() {}

  // Image information
  int Rows() const;
  int Cols() const;
  int Width() const;
  int Height() const;
  int Channels() const;

  // Set the pixel color value at (x, y) in channel c.
  void SetXY(const int x, const int y, const int c, const T value);
  // Set the rgb color values at the pixel (x, y). This assumes (with no
  // checks!) that the image is an rgb image.
  void SetXY(const int x, const int y, const Eigen::Matrix<T, 3, 1>& rgb);

  // Get the pixel value at the given (x, y) position and channel.
  T GetXY(const int x, const int y, const int c) const;
  // Get the RGB value at the given pixel. This assumes (with no checks!) that
  // the image is indeed an RGB image.
  Eigen::Matrix<T, 3, 1> GetXY(const int x, const int y) const;

  // Set the pixel value at row and column in channel c.
  void SetRowCol(const int row,
                 const int col,
                 const int channel,
                 const T value);
  // Set the rgb color values at the given row and column. This assumes (with no
  // checks!) that the image is an rgb image.
  void SetRowCol(const int row,
                 const int col,
                 const Eigen::Matrix<T, 3, 1>& value);
  // Get the pixel value at the given location and channel.
  T GetRowCol(const int row, const int col, const int channel) const;

  // Get the RGB value at the given pixel. This assumes (with no checks!) that
  // the image is indeed an RGB image.
  Eigen::Matrix<T, 3, 1> GetRowCol(const int row, const int col) const;

  // Get the pixel value at a non-discrete location with bilinear interpolation.
  T BilinearInterpolate(const double x, const double y, const int c = 0) const;

  // Get the pixel value at a non-discrete location with bilinear interpolation.
  Eigen::Matrix<T, 3, 1> BilinearInterpolate(const double x, const double y) const;

  // Convert to other image types.
  void ConvertToGrayscaleImage();
  void ConvertToRGBImage();
  Image<T> AsGrayscaleImage() const;
  Image<T> AsRGBImage() const;

  // Scale all the pixel values by a scale factor.
  void ScalePixels(float scale);

  // Write image to file.
  void Read(const std::string& filename);
  void Write(const std::string& filename) const;

  // Get a pointer to the data.
  T* Data() { return image_.data(); }
  const T* Data() const { return image_.data(); }

  // Computes the gradient in each respective dimension.
  Image<T> ComputeGradientX() const;
  Image<T> ComputeGradientY() const;
  // Computes the gradient in x and y and returns the summation to obtain the
  // gradient magnitude at each pixel.
  Image<T> ComputeGradient() const;

  // Sampling techniques.
  void HalfSample(Image<T>* dest) const;
  void TwoThirdsSample(Image<T>* dest) const;

  // Apply a median filter to the image. Each pixel value is replaced by taking
  // the median value in a patch_width x patch_width window centered at the
  // pixel.
  void MedianFilter(const int patch_width);

  // Compute the integral image where pixel (x, y) is equal to the sum of all
  // values in the rectangle from (0, 0) to (x, y) non-inclusive. This means
  // that the first row and column are all zeros, and the returned integral
  // image is one pixel wider and taller than the caller.
  //
  // NOTE: This method should be called with precise number types such as double
  // otherwise floating roundoff errors are sure to occur.
  template <typename D>
  void Integrate(Image<D>* integral) const;

  // Computes a fast approximate gaussian blur of te image.
  void ApproximateGaussianBlur(const double sigma);

  // Resize using a Lanczos 3 filter.
  void Resize(int new_width, int new_height);
  void Resize(int new_width, int new_height, int new_channels);
  void ResizeRowsCols(int new_rows, int new_cols);
  void ResizeRowsCols(int new_rows, int new_cols, int new_channels);
  void Resize(double scale);

 protected:
  template<class AnyType> friend class Image;
  friend class ImageCanvas;

  cimg_library::CImg<T> image_;
};

// ----------------- Implementation ----------------- //

template <typename T> Image<T>::Image() : image_(1, 1, 1, 1) {}

// Read from file.
template <typename T> Image<T>::Image(const std::string& filename) {
  image_.load(filename.c_str());
}

template <typename T> Image<T>::Image(const Image<T>& image_to_copy) {
  image_ = image_to_copy.image_;
}

template <typename T>
Image<T>::Image(const int width, const int height, const int channels) {
  image_.resize(width, height, 1, channels);
}

// Copy function. This is a deep copy of the image.
template <typename T> template <typename D>
Image<T>::Image(const Image<D>& image_to_copy) {
  image_ = image_to_copy.image_;
}

template <typename T> Image<T>::Image(const cimg_library::CImg<T>& image) {
  image_ = image;
}

template <typename T> int Image<T>::Rows() const {
  return image_.height();
}

template <typename T> int Image<T>::Cols() const {
  return image_.width();
}

template <typename T> int Image<T>::Width() const {
  return image_.width();
}

template <typename T> int Image<T>::Height() const {
  return image_.height();
}

template <typename T> int Image<T>::Channels() const {
  return image_.spectrum();
}

template <typename T>
void Image<T>::SetXY(const int x, const int y, const int c, const T value) {
  image_(x, y, 0, c) = value;
}

template <typename T>
void Image<T>::SetXY(const int x, const int y, const Eigen::Matrix<T, 3, 1>& rgb) {
  image_(x, y, 0, 0) = rgb(0);
  image_(x, y, 0, 1) = rgb(1);
  image_(x, y, 0, 2) = rgb(2);
}

template <typename T>
T Image<T>::GetXY(const int x, const int y, const int c) const {
  return image_(x, y, 0, c);
}

template <typename T>
Eigen::Matrix<T, 3, 1> Image<T>::GetXY(const int x, const int y) const {
  Eigen::Matrix<T, 3, 1> out;
  out(0) = image_(x, y, 0, 0);
  out(1) = image_(x, y, 0, 1);
  out(2) = image_(x, y, 0, 2);
  return out;
};

template <typename T>
void Image<T>::SetRowCol(const int row, const int col, const int channel, const T value) {
  image_(col, row, channel) = value;
}

template <typename T>
void Image<T>::SetRowCol(const int row, const int col, const Eigen::Matrix<T, 3, 1>& value) {
  image_(col, row, 0, 0) = value(0);
  image_(col, row, 0, 1) = value(1);
  image_(col, row, 0, 2) = value(2);
}

template <typename T>
T Image<T>::GetRowCol(const int row, const int col, const int channel) const {
  return image_(col, row, channel);
}

template <typename T>
Eigen::Matrix<T, 3, 1> Image<T>::GetRowCol(const int row, const int col) const {
  Eigen::Matrix<T, 3, 1> out;
  out(0) = image_(col, row, 0, 0);
  out(1) = image_(col, row, 0, 1);
  out(2) = image_(col, row, 0, 2);
  return out;
}

template <typename T>
T Image<T>::BilinearInterpolate(const double x, const double y, const int c) const {
  return image_.linear_atXY(x, y, 0, c);
}

template <typename T>
Eigen::Matrix<T, 3, 1> Image<T>::BilinearInterpolate(const double x, const double y) const {
  Eigen::Matrix<T, 3, 1> out;
  out(0) = image_.linear_atXY(x, y, 0, 0);
  out(1) = image_.linear_atXY(x, y, 0, 1);
  out(2) = image_.linear_atXY(x, y, 0, 2);
  return out;
}

template <typename T> void Image<T>::ConvertToGrayscaleImage() {
  if (Channels() == 1) {
    VLOG(2) << "Image is already a grayscale image. No conversion necessary.";
    return;
  }
  image_ = image_.get_RGBtoYCbCr().channel(0);
}

template <typename T> void Image<T>::ConvertToRGBImage() {
  if (Channels() == 3) {
    VLOG(2) << "Image is already an RGB image. No conversion necessary.";
    return;
  }

  // Resizing with NN interpolation by default will copy the pixel value of the
  // grayscale pixel to all RGB channels.
  image_.resize(image_.width(),
                image_.height(),
                image_.depth(),
                3);
}

template <typename T> Image<T> Image<T>::AsGrayscaleImage() const {
  if (Channels() == 1) {
    VLOG(2) << "Image is already a grayscale image. No conversion necessary.";
    return *this;
  }
  Image<T> gray_image(*this);
  gray_image.ConvertToGrayscaleImage();
  return gray_image;
}

template <typename T> Image<T> Image<T>::AsRGBImage() const {
  if (Channels() == 3) {
    VLOG(2) << "Image is already an RGB image. No conversion necessary.";
    return *this;
  }

  Image<T> rgb_image(*this);
  rgb_image.ConvertToRGBImage();
  return rgb_image;
}

template <typename T> void Image<T>::ScalePixels(float scale) {
  image_ *= scale;
}

template <typename T> void Image<T>::Read(const std::string& filename) {
  image_.load(filename.c_str());
}

template <typename T> void Image<T>::Write(const std::string& filename) const {
  image_.save(filename.c_str());
}

template <typename T>
Image<T> Image<T>::ComputeGradientX() const {
  return Image<T>(image_.get_gradient("x", 2)[0]);
}

template <typename T>
Image<T> Image<T>::ComputeGradientY() const {
  return Image<T>(image_.get_gradient("y", 2)[0]);
}

template <typename T>
Image<T> Image<T>::ComputeGradient() const {
  cimg_library::CImgList<T> gradient = image_.get_gradient("xy", 2);
  return Image<T>(gradient[0] + gradient[1]);
}

template <typename T> void Image<T>::HalfSample(Image<T>* dest) const {
  dest->image_ = image_.get_resize_halfXY();
}

template <typename T> void Image<T>::TwoThirdsSample(Image<T>* dest) const {
  const int new_width = static_cast<int>(image_.width() * 2.0 / 3.0);
  const int new_height = static_cast<int>(image_.height() * 2.0 / 3.0);
  dest->image_ = image_.get_resize(new_width, new_height, -100, -100, 6);
}

template <typename T>
void Image<T>::MedianFilter(const int patch_width) {
  image_ = image_.blur_median(patch_width);
}

template <typename T>
template <typename D>
void Image<T>::Integrate(Image<D>* integral) const {
  integral->ResizeRowsCols(Rows() + 1, Cols() + 1);

  for (int i = 0; i < Channels(); i++) {
    // Fill the first row with zeros.
    for (int x = 0; x < Width(); x++) {
      (*integral).SetXY(x, 0, i, 0);
    }

    for (int y = 1; y <= Height(); y++) {
      // This variable is to correct floating point round off.
      D sum(0);
      (*integral).SetXY(0, y, i, 0);
      for (int x = 1; x <= Width(); x++) {
        sum += static_cast<D>((*this).GetXY(x - 1, y - 1, i));
        (*integral).SetXY(x, y, i, (*integral).GetXY(x, y - 1, i) + sum);
      }
    }
  }
}

template <typename T>
void Image<T>::ApproximateGaussianBlur(const double sigma) {
  image_.blur(sigma);
}

template <typename T>
void Image<T>::Resize(int new_width, int new_height) {
  image_.resize(new_width, new_height);
}

template <typename T>
void Image<T>::Resize(int new_width, int new_height, int new_channels) {
  image_.resize(new_width, new_height, 1, new_channels);
}

template <typename T>
void Image<T>::ResizeRowsCols(int new_rows, int new_cols) {
  image_.resize(new_cols, new_rows);
}
template <typename T>
void Image<T>::ResizeRowsCols(int new_rows, int new_cols, int new_channels) {
  image_.resize(new_cols, new_rows, 1, new_channels);
}

template <typename T>
void Image<T>::Resize(double scale) {
  int new_width = scale * image_.width();
  int new_height = scale * image_.height();
  this->Resize(new_width, new_height);
}

}  // namespace theia

#endif  // THEIA_IMAGE_IMAGE_H_
