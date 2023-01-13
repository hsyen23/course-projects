# Foreground/background segmentation
Perform image segmentation via statistical learning. 

Use fft coefficients as features to create gaussian distributions for background and foregound on training data. Then, classify pixels in target image by the distributions.

Original image

![Alt text](PR1/pic/cheetah.jpg "cheetah")

Ground truth

![Alt text](PR1/pic/cheetah_mask.jpg "cheetah_mask")

## PR1: Second largest representative factor
Use second largest fft coefficient as the feature to do segmentation.

Result

![Alt text](PR1/pic/result.jpg "result")

## PR3,4: 

Result
### x-axis: alpha y-axis: error

![Alt text](PR3,4/pic/HW3_comparision1.JPG "HW3_comparision1")

![Alt text](PR3,4/pic/HW3_comparision2.JPG "HW3_comparision2")

