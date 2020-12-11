# SuperResolutionusingOpenCVandDeepLearning

Super Resolution(SR)
1.  More simply, take an input image and increase the width and height of the image with minimal (and ideally zero) degradation in quality.
2.  Photoshop, GIMP, Image Magick, OpenCV (via the cv2.resize function), etc. all use classic interpolation techniques and algorithms (ex., nearest neighbor interpolation, linear interpolation, bicubic interpolation) to increase the image resolution.
3. By applying novel deep learning architectures, we’re able to generate high resolution images without these artifacts

Check this super resolution models : https://www.pyimagesearch.com/wp-content/uploads/2020/11/opencv_super_resolution_fsrcnn_arch.png

Check this SR Model implementation papers :

EDSR: Enhanced Deep Residual Networks for Single Image Super-Resolution (implementation)
ESPCN: Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network (implementation)
FSRCNN: Accelerating the Super-Resolution Convolutional Neural Network (implementation)
LapSRN: Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks (implementation)


Using Above mentioned models to create the super resolution model and it wil incease the width and height of the image with minimal (and ideally zero) degradation in quality.