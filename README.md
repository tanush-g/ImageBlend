# ImageBlend

This project aims to develop a CLI interface to blend two input images by face swapping. The motivation arises from the common issue of capturing perfect photos, where often the subject's eyes are closed, they are not smiling correctly, or other issues arise. This project provides a solution by allowing users to swap faces from different images, preserving the desired facial expressions and features.

## Overview

The project utilizes image blending techniques, Gaussian Pyramids, Laplacian Pyramids, facial detection, and feature matching algorithms (ORB and CascadeClassifier) to seamlessly blend two images while preserving the principal parts of the overlapping images.

## Theory

![Image Pyramid](https://upload.wikimedia.org/wikipedia/commons/4/43/Image_pyramid.svg)
Cmglee, CC BY-SA 3.0 <https://creativecommons.org/licenses/by-sa/3.0>, via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Image_pyramid.svg)

### Gaussian Pyramid

The Gaussian Pyramid is a technique in image processing that breaks down an image into successively smaller groups of pixels to blur it. This process enables edge detection, making it easier for computers to identify objects automatically. The pyramid is constructed by repeatedly calculating a weighted average of the neighboring pixels and scaling down the image.

### Laplacian Pyramid

The Laplacian Pyramid is a bandpass image decomposition derived from the Gaussian Pyramid. It is a multiresolution image representation obtained through a recursive reduction (lowpass filtering and decimation) of the image dataset. The image is recursively lowpass filtered and downsampled to generate a lowpass sub-band, which is re-expanded and subtracted pixel-by-pixel from the original image to yield the 2-D detail image with zero-mean.

### Feature Selection using ORB and CascadeClassifier

The project uses ORB (Oriented FAST and Rotated BRIEF) for feature selection and CascadeClassifier for detecting faces in digital images to generate the mask. ORB is a fusion of the FAST keypoint detector and BRIEF descriptor with modifications to enhance performance. It uses FAST to find keypoints, applies the Harris corner measure to find the top N points among them, and employs pyramids to produce multiscale features.

Object Detection using Haar feature-based cascade classifiers is a machine learning-based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.

## Methodology

1. Construct Laplacian Pyramids for the source and target images' channels.
2. Extract features for face detection to generate the required binary mask, where 1 signifies the overlap of the first image pixel on the second one, and 0 ignores the second image pixel value.
3. Blend the two pyramids based on a corresponding Gaussian Pyramid for the mask.
4. Reconstruct the blended image by combining the levels of the blended Laplacian Pyramid.

The project optimizes the computation by convoluting with the kernel in the Fourier domain and then taking an inverse Fourier transform to obtain the resultant image.

## Installation

To install and set up the project, follow these steps:

1. Clone the repository: `git clone https://github.com/tanush-g/imageblend.git`

2. Navigate to the project directory: `cd imageblend`

3. Install the required dependencies: `pip install -r requirements.txt`

## Usage

To run the algorithm, execute the following command in the directory containing the code and images:

`python source.py`

You will be prompted to enter the names of the two input images (with file extensions). The blended image will be displayed and saved as `result.jpg`.

## Contributing

Contributions to the project are welcome! If you find any issues or want to add new features, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive commit messages.
4. Push your changes to your forked repository.
5. Submit a pull request to the main repository.

Please ensure that your code adheres to the project's code style guidelines and that you include appropriate tests for any new features or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).

## Results

The project has been tested with various input images, including scenarios where faces are not smiling correctly or eyes are closed. The algorithm effectively blends the faces from different images, preserving the desired facial expressions and features.

## Conclusion

The project successfully accomplishes the task of blending two images of people to obtain an image where the face swap is nearly unnoticeable to the naked eye. Several adaptations have been made to optimize the code's running time, such as shifting the Gaussian Pyramid Level computation to the Fourier Domain.

## References

The project report includes references to relevant papers and resources used in the development of this project.
