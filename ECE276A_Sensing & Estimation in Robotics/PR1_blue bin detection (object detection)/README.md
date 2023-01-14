# Blue-bin detector via logistic regression
Develop a blue-bin detector via logistic regression on color classification and morphological properties of detected blue regions.
## Process
1. identify blue region (color of recycling bin)

2. image processing (erosion and dilation)

3. use bounding box to do annotation

4. create the mask for detected bin

## ex1
target:

![Alt text](pic/0001/0001.jpg "0001")

blue extraction:

![Alt text](pic/0001/_original.png "0001_blue_extraction")

image processing:

![Alt text](pic/0001/_erosion_2.png "0001_erosion")

bounding box:

![Alt text](pic/0001/_dialation_1.png "0001_box")

mask:

![Alt text](pic/0001/_another_mask.png  "0001_mask")

## ex2
target:

![Alt text](pic/0003/0003.jpg "0003")

blue extraction:

![Alt text](pic/0003/_original.png "0003_blue_extraction")

image processing:

![Alt text](pic/0003/_erosion_2.png "0003_erosion")

bounding box:

![Alt text](pic/0003/_dialation_1.png "0003_box")

mask:

![Alt text](pic/0003/_another_mask.png  "0003_mask")

## ex3
target:

![Alt text](pic/0010/0010.jpg "0010")

blue extraction:

![Alt text](pic/0010/_original.png "0010_blue_extraction")

image processing:

![Alt text](pic/0010/_erosion_2.png "0010_erosion")

bounding box:

![Alt text](pic/0010/_dialation_1.png "0010_box")

mask:

![Alt text](pic/0010/_another_mask.png  "0010_mask")
