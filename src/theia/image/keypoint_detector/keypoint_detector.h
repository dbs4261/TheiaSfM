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

#ifndef THEIA_IMAGE_KEYPOINT_DETECTOR_KEYPOINT_DETECTOR_H_
#define THEIA_IMAGE_KEYPOINT_DETECTOR_KEYPOINT_DETECTOR_H_

#include <vector>

#include "theia/image/keypoint_detector/keypoint.h"
#include "theia/util/util.h"

namespace theia {
template <typename T> class Image;
typedef Image<float> FloatImage;

// A pure virtual class for keypoint detectors. We assume that the keypoint
// detectors only use grayimages for now.
class KeypointDetector {
 public:
  KeypointDetector() {}
  virtual ~KeypointDetector() {}

  // Use this method to initialize any internals. Only use the constructor for
  // basic operations since the debug trace is limited for errors in the
  // constructor.
  virtual bool Initialize() { return true; }

  // Detect keypoints using the desired method. This method will allocate the
  // Keypoint pointers in the vector with new, but the caller owns the data
  // returned (and must delete the pointers).
  virtual bool DetectKeypoints(const FloatImage& image,
                               std::vector<Keypoint>* keypoints) = 0;

 private:
  DISALLOW_COPY_AND_ASSIGN(KeypointDetector);
};

}  // namespace theia

#endif  // THEIA_IMAGE_KEYPOINT_DETECTOR_KEYPOINT_DETECTOR_H_
