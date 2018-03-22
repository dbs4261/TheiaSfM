// Copyright (C) 2013 The Regents of the University of California (Regents).
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

#include <cimg/CImg.h>
#include <Eigen/Core>
#include <gflags/gflags.h>
#include <stdio.h>
#include <string>

#include "gtest/gtest.h"
#include "theia/image/image.h"
#include "theia/util/random.h"

DEFINE_string(test_img, "image/test1.jpg", "Name of test image file.");

namespace theia {
namespace {
using cimg_library::CImg;

RandomNumberGenerator rng(51);

std::string img_filename = THEIA_DATA_DIR + std::string("/") + FLAGS_test_img;

#define ASSERT_IMG_EQ(cimg_img, theia_img)                            \
  ASSERT_EQ((cimg_img).width(), (theia_img).Width());                 \
  ASSERT_EQ((cimg_img).height(), (theia_img).Height());               \
  ASSERT_EQ((cimg_img).spectrum(), (theia_img).Channels());           \
  ASSERT_EQ((cimg_img).depth(), 1);                                   \
  for (int y = 0; y < (theia_img).Height(); y++) {                      \
    for (int x = 0; x < (theia_img).Width(); x++) {                    \
      for (int d = 0; d < (theia_img).Channels(); d++) {              \
        ASSERT_EQ((cimg_img)(x, y, 0, d), (theia_img).GetXY(x, y, d)) \
        << "Failure at " << x << ", " << y << ", " << d;              \
      }                                                               \
    }                                                                 \
  }

float Interpolate(const FloatImage& image,
                  const float x,
                  const float y,
                  const int c) {
  const auto left = (int)std::floor(x);
  const auto right = (int)std::ceil(x);
  const auto top = (int)std::floor(y);
  const auto bottom = (int)std::ceil(y);
  return image.GetXY(left, top, c) * (right - x) * (bottom - y) +
         image.GetXY(left, bottom, c) * (right - x) * (y - top) +
         image.GetXY(right, top, c) * (x - left) * (bottom - y) +
         image.GetXY(right, bottom, c) * (x - left) * (y - top);
}

}  // namespace

// Test that inputting the old fashioned way is the same as through our class.
TEST(Image, RGBInput) {
  CImg<float> cimg_img(img_filename.c_str());
  FloatImage theia_img(img_filename);

  ASSERT_EQ(theia_img.Width(), 1024);
  ASSERT_EQ(theia_img.Height(), 679);
  ASSERT_EQ(theia_img.Channels(), 3);
  // Assert each pixel value is exactly the same!
  ASSERT_IMG_EQ(cimg_img, theia_img);
}

// Test that width and height methods work.
TEST(Image, RGBColsRows) {
  CImg<float> cimg_img(img_filename.c_str());
  FloatImage theia_img(img_filename);

  int true_height = cimg_img.height();
  int true_width = cimg_img.width();

  ASSERT_EQ(theia_img.Cols(), true_width);
  ASSERT_EQ(theia_img.Rows(), true_height);
}

// Test that inputting the old fashioned way is the same as through our class.
TEST(Image, ConvertToGrayscaleImage) {
  CImg<float> cimg_img(img_filename.c_str());
  CImg<float> gray_img(cimg_img.RGBtoYCbCr().channel(0));
  FloatImage theia_img(img_filename);
  ASSERT_EQ(theia_img.Channels(), 3);
  theia_img.ConvertToGrayscaleImage();
  ASSERT_EQ(theia_img.Channels(), 1);

  int rows = cimg_img.height();
  int cols = cimg_img.width();

  // Assert each pixel value is exactly the same!
  ASSERT_IMG_EQ(gray_img, theia_img);
}

TEST(Image, ConvertToRGBImage) {
  const CImg<float> cimg_img(img_filename.c_str());
  const CImg<float>& gray_cimg = cimg_img.get_RGBtoYCbCr().get_channel(0);

  CImg<float> rgb_img = gray_cimg;
  rgb_img.resize(cimg_img.width(), cimg_img.height(), cimg_img.depth(),
                 3);

  cimg_forXY(rgb_img, x, y) {
    CHECK_EQ(rgb_img(x, y, 0, 0), rgb_img(x, y, 0, 1));
    CHECK_EQ(rgb_img(x, y, 0, 0), rgb_img(x, y, 0, 2));
  }

  FloatImage theia_img(img_filename);
  theia_img.ConvertToGrayscaleImage();
  theia_img.ConvertToRGBImage();

  int rows = cimg_img.height();
  int cols = cimg_img.width();

  // Assert each pixel value is exactly the same!
  ASSERT_IMG_EQ(rgb_img, theia_img);
}

TEST(Image, IntegralImage) {
  const FloatImage img = FloatImage(img_filename).AsGrayscaleImage();
  Image<double> integral_img;
  img.Integrate(&integral_img);

  // Check the integral image over 100 trials;
  for (int i = 0; i < 1000; i++) {
    const int test_col = rng.RandInt(1, img.Cols());
    const int test_row = rng.RandInt(1, img.Rows());

    // Check the integral.
    double sum = 0;
    for (int r = 0; r < test_row; r++) {
      for (int c = 0; c < test_col; c++) {
        sum += img.GetXY(c, r, 0);
      }
    }

    EXPECT_DOUBLE_EQ(integral_img.GetXY(test_col, test_row, 0), sum);
  }
}

TEST(Image, BillinearInterpolate) {
  static const int kNumTrials = 10;
  static const float kTolerance = 1e-2;

  FloatImage theia_img(img_filename);
  theia_img.ConvertToGrayscaleImage();
  for (int i = 0; i < kNumTrials; i++) {
    const double x = rng.RandDouble(1.0, theia_img.Width() - 2);
    const double y = rng.RandDouble(1.0, theia_img.Height() - 2);
    const float pixel = Interpolate(theia_img, x, y, 0);
    const float pixel2 = theia_img.BilinearInterpolate(x, y, 0);
    EXPECT_NEAR(pixel, pixel2, kTolerance);
  }
}

TEST(Image, ScalePixels) {
  static const float kTolerance = 1e-2;
  static const float kScaleFactor = 1.1;

  FloatImage theia_img(img_filename);
  FloatImage scaled_img(img_filename);
  scaled_img.ScalePixels(kScaleFactor);

  for (int y = 0; y < theia_img.Height(); y++) {
    for (int x = 0; x < theia_img.Width(); x++) {
      ASSERT_EQ(kScaleFactor * theia_img.GetXY(x, y, 0),
                scaled_img.GetXY(x, y, 0));
    }
  }
}

TEST(Image, Resize) {
  static const int kWidth = 800;
  static const int kHeight = 600;

  FloatImage theia_img(img_filename);
  theia_img.Resize(kWidth, kHeight);

  // Make sure the image was resized appropriately.
  EXPECT_EQ(theia_img.Width(), kWidth);
  EXPECT_EQ(theia_img.Height(), kHeight);
}

TEST(Image, ResizeUninitialized) {
  static const int kWidth = 800;
  static const int kHeight = 600;

  FloatImage theia_img;
  theia_img.Resize(kWidth, kHeight);

  // Make sure the image was resized appropriately.
  EXPECT_EQ(theia_img.Width(), kWidth);
  EXPECT_EQ(theia_img.Height(), kHeight);

  // Make sure the resizing still works when converting an uninitialized image
  // to RGB first.
  FloatImage theia_img2;
  theia_img2.ConvertToRGBImage();
  theia_img2.Resize(kWidth, kHeight);

  // Make sure the image was resized appropriately.
  EXPECT_EQ(theia_img2.Width(), kWidth);
  EXPECT_EQ(theia_img2.Height(), kHeight);
  EXPECT_EQ(theia_img2.Channels(), 3);
}

}  // namespace theia
