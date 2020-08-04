# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from PIL import Image
from pose_engine import PoseEngine
import cv2
#os.system('wget https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/'
          #'Hindu_marriage_ceremony_offering.jpg/'
          #'640px-Hindu_marriage_ceremony_offering.jpg -O couple.jpg')
pil_image = Image.open('couple.jpg')
pil_image.resize((1281, 721), Image.NEAREST)
cvim=np.asarray(pil_image)
engine = PoseEngine('models/posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite')
poses, inference_time = engine.DetectPosesInImage(np.uint8(pil_image))
print('Inference time: %.fms' % inference_time)
#cv2.imshow("original",cvim)
cvim=cvim[:,:,::-1].copy()
i=0
for pose in poses:
    if pose.score < 0.4: continue
    j=0
    print('\nPose Score: ', pose.score)
    for label, keypoint in pose.keypoints.items():
        print(' %-20s x=%-4d y=%-4d score=%.1f' %
              (label, keypoint.yx[1], keypoint.yx[0], keypoint.score))
        k=keypoint.score
        cv2.circle(cvim,(keypoint.yx[1],keypoint.yx[0]),4,(int(k*i*255),int(k*j*255/16),int(255*k)),-1)
        j+=1
    i+=1
cv2.imshow("inference",cvim)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("coupleoutputL.jpg", cvim)
