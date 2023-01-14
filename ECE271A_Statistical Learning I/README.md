# Foreground/background segmentation
Perform image segmentation via statistical learning. 

Use fft coefficients as features to create gaussian distributions for background and foregound on training data. Then, classify pixels in target image by the distributions.

Original image

![Alt text](PR1/pic/cheetah.jpg "cheetah")

Ground truth

![Alt text](PR1/pic/cheetah_mask.jpg "cheetah_mask")

## PR1: Second largest representative factor
Use second largest fft coefficient as the feature to do segmentation because the largest one is always DC signal.

Result:

![Alt text](PR1/pic/result_error=0.1816.jpg "result_error=0.1816")

Error rate: 0.1816

## PR2: Multivariate features (64d and 8d)
Use all fft coefficitents (64d) to build classifier, then select top eight representative coefficients (8d) to build classifier.

64d result:
![Alt text](PR2/pic/result_64d_error=0.094.jpg "result_64d_error=0.094")

Error rate: 0.094

8d result:
![Alt text](PR2/pic/result_8d_error=0.063.jpg "result_8d_error=0.063")

Error rate: 0.063


## PR3,4: 

Result
### x-axis: alpha y-axis: error

![Alt text](PR3,4/pic/HW3_comparision1.JPG "HW3_comparision1")

![Alt text](PR3,4/pic/HW3_comparision2.JPG "HW3_comparision2")

