import cv2
import numpy as np
from numpy.fft import ifft2, fft2
import matplotlib.pyplot as plt
from copy import deepcopy
import math
import sys # Keep sys for sys.exit

# --- Constants ---
PYRAMID_DEPTH_ADJUST = 4
FACE_DETECT_SCALE_FACTOR = 1.3
FACE_DETECT_MIN_NEIGHBORS = 5
MASK_RADIUS_MULTIPLIER = 1.2
FEATURE_MATCH_DISPLAY_COUNT = 50
RANSAC_REPROJ_THRESHOLD = 5.0
OUTPUT_FILENAME = 'result.jpg'
DISPLAY_RESIZE_FACTOR = 0.5
GAUSS_KERNEL_VALUES = np.array([0.05, 0.25, 0.4, 0.25, 0.05])
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# ---

# Global face cascade variable (initialized in main)
face_cascade = None

def build_gaussian_pyramid(image, levels):
    """Constructs a Gaussian pyramid for an image.

    Args:
        image (np.ndarray): The input image (single channel, float).
        levels (int): The number of pyramid levels to build.

    Returns:
        list[np.ndarray]: A list of images forming the Gaussian pyramid.
    """
    output = [image]
    current_image = image
    # Use a pre-defined kernel
    kernel = np.outer(GAUSS_KERNEL_VALUES, GAUSS_KERNEL_VALUES)
    fft_kernel_cache = {} # Cache FFT of kernel for different shapes

    for _ in range(levels):
        # Convolution via FFT saves immense time
        if current_image.shape not in fft_kernel_cache:
            fft_kernel_cache[current_image.shape] = fft2(kernel, s=current_image.shape)
        fft_kernel = fft_kernel_cache[current_image.shape]

        blurred_image = np.abs(ifft2(np.multiply(fft2(current_image), fft_kernel)))

        # Reducing by a factor of 2 (downsampling)
        downsampled_image = blurred_image[::2, ::2]
        output.append(downsampled_image)
        current_image = downsampled_image
    return output

def expand_image(image):
    """Expands an image to twice its size using Gaussian blurring (inverse of pyramid reduction).

    Args:
        image (np.ndarray): The input image (single channel, float).

    Returns:
        np.ndarray: The expanded image.
    """
    rows, cols = image.shape
    expanded_image = np.zeros((2 * rows, 2 * cols), dtype=np.float64)
    expanded_image[::2, ::2] = image[:, :]

    # Use a pre-defined kernel
    kernel = np.outer(GAUSS_KERNEL_VALUES, GAUSS_KERNEL_VALUES)
    # Convolution via FFT
    fft_kernel = fft2(kernel, s=expanded_image.shape)
    convolved_image = np.abs(ifft2(np.multiply(fft2(expanded_image), fft_kernel)))

    # Multiply by 4 to compensate for energy loss during expansion
    return 4 * convolved_image

def build_laplacian_pyramid(gaussian_pyramid):
    """Constructs a Laplacian pyramid from a Gaussian pyramid.

    Args:
        gaussian_pyramid (list[np.ndarray]): The input Gaussian pyramid.

    Returns:
        list[np.ndarray]: The Laplacian pyramid.
    """
    output = []
    num_levels = len(gaussian_pyramid)

    for i in range(num_levels - 1):
        current_level = gaussian_pyramid[i]
        rows, cols = current_level.shape
        expanded_next_level = expand_image(gaussian_pyramid[i + 1])
        rows_expanded, cols_expanded = expanded_next_level.shape

        # Adjust dimensions if they differ due to odd sizes during downsampling
        if rows_expanded > rows:
            expanded_next_level = np.delete(expanded_next_level, (-1), axis=0)
        if cols_expanded > cols:
            expanded_next_level = np.delete(expanded_next_level, (-1), axis=1)

        # Laplacian level is the difference
        output.append(current_level - expanded_next_level)

    # The last level of the Laplacian pyramid is the smallest image from the Gaussian pyramid
    output.append(gaussian_pyramid[-1])
    return output

def blend_pyramids(laplacian_pyr1, laplacian_pyr2, mask_gaussian_pyr):
    """Blends two Laplacian pyramids using a Gaussian mask pyramid.

    Args:
        laplacian_pyr1 (list[np.ndarray]): Laplacian pyramid of the first image.
        laplacian_pyr2 (list[np.ndarray]): Laplacian pyramid of the second image.
        mask_gaussian_pyr (list[np.ndarray]): Gaussian pyramid of the mask.

    Returns:
        list[np.ndarray]: The blended Laplacian pyramid.
    """
    blended_pyramid = []
    for i in range(len(mask_gaussian_pyr)):
        mask_level = mask_gaussian_pyr[i]
        level1 = laplacian_pyr1[i]
        level2 = laplacian_pyr2[i]
        # Ensure mask level has same dimensions as laplacian levels
        rows, cols = level1.shape
        mask_level = cv2.resize(mask_level, (cols, rows)) # Resize mask if needed
        blended_level = (mask_level * level1) + ((1 - mask_level) * level2)
        blended_pyramid.append(blended_level)
    return blended_pyramid

def reconstruct_from_laplacian_pyramid(laplacian_pyr):
    """Reconstructs an image from its Laplacian pyramid.

    Args:
        laplacian_pyr (list[np.ndarray]): The Laplacian pyramid.

    Returns:
        np.ndarray: The reconstructed image.
    """
    # Start with the smallest level (the residual)
    current_image = laplacian_pyr[-1]
    num_levels = len(laplacian_pyr)

    for i in range(num_levels - 2, -1, -1): # Iterate backwards from second-to-last level
        expanded_image = expand_image(current_image)
        laplacian_level = laplacian_pyr[i]
        rows, cols = laplacian_level.shape
        rows_expanded, cols_expanded = expanded_image.shape

        # Adjust dimensions if necessary
        if rows_expanded > rows:
            expanded_image = np.delete(expanded_image, (-1), axis=0)
        if cols_expanded > cols:
            expanded_image = np.delete(expanded_image, (-1), axis=1)

        # Add the expanded lower level to the current Laplacian level
        current_image = expanded_image + laplacian_level

    return current_image

def blend_images(target_image, source_image, mask):
    """Performs Laplacian pyramid blending of two images using a mask.

    Args:
        target_image (np.ndarray): The background image (BGR, uint8).
        source_image (np.ndarray): The foreground image (BGR, uint8) to blend onto the target.
        mask (np.ndarray): A mask (grayscale or BGR, uint8) where white areas indicate
                           regions of the source image to use.

    Returns:
        np.ndarray: The blended image (BGR, uint8).
    """
    # Ensure mask is 3-channel if it isn't already
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Split channels and convert to float
    (b1, g1, r1) = cv2.split(target_image)
    target_channels = [b1.astype(float), g1.astype(float), r1.astype(float)]

    (b2, g2, r2) = cv2.split(source_image)
    source_channels = [b2.astype(float), g2.astype(float), r2.astype(float)]

    # Ensure mask is float and normalized [0, 1]
    (bm, gm, rm) = cv2.split(mask)
    mask_channels = [(bm.astype(float) / 255.0), (gm.astype(float) / 255.0), (rm.astype(float) / 255.0)]

    # Decide the number of pyramid levels
    min_dim = min(target_channels[0].shape[0], target_channels[0].shape[1])
    depth = int(math.floor(math.log(min_dim, 2))) - PYRAMID_DEPTH_ADJUST
    if depth < 1:
        print("Warning: Image too small for specified pyramid depth adjustment. Using depth=1.")
        depth = 1

    blended_channels_uint8 = []

    for i in range(3): # Process R, G, B channels
        # Build Gaussian pyramid for the mask channel
        mask_gauss_pyr = build_gaussian_pyramid(mask_channels[i], depth)

        # Build Laplacian pyramids for the image channels
        target_gauss_pyr = build_gaussian_pyramid(target_channels[i], depth)
        laplacian_pyr_target = build_laplacian_pyramid(target_gauss_pyr)

        source_gauss_pyr = build_gaussian_pyramid(source_channels[i], depth)
        laplacian_pyr_source = build_laplacian_pyramid(source_gauss_pyr)

        # Blend the Laplacian pyramids
        # Note: Order matters - blend source *onto* target based on mask
        blended_laplacian_pyr = blend_pyramids(laplacian_pyr_source, laplacian_pyr_target, mask_gauss_pyr)

        # Reconstruct the blended channel
        blended_channel = reconstruct_from_laplacian_pyramid(blended_laplacian_pyr)

        # Clamp values and convert back to uint8
        blended_channel = np.clip(blended_channel, 0, 255)
        blended_channels_uint8.append(blended_channel.astype(np.uint8))

    # Merge the channels back
    # Order is Blue, Green, Red for cv2.merge
    result_image = cv2.merge((blended_channels_uint8[0], blended_channels_uint8[1], blended_channels_uint8[2]))

    # Save and display result
    cv2.imwrite(OUTPUT_FILENAME, result_image)
    print(f"Blended image saved as {OUTPUT_FILENAME}")

    # Reduce dimensions for viewing purposes
    display_width = int(result_image.shape[1] * DISPLAY_RESIZE_FACTOR)
    display_height = int(result_image.shape[0] * DISPLAY_RESIZE_FACTOR)
    display_dims = (display_width, display_height)
    result_display = cv2.resize(result_image, display_dims, interpolation=cv2.INTER_AREA)
    cv2.imshow("Blended Image", result_display)
    # cv2.waitKey(0) # Keep commented unless interactive display is needed

    return result_image # Return the full-resolution blended image

def detect_faces(image):
    """Detects faces in an image using the loaded Haar cascade.

    Args:
        image (np.ndarray): The input image (BGR, uint8).

    Returns:
        tuple: A tuple containing:
            - list[float]: List of face radii.
            - list[list[float]]: List of face center coordinates [[x1, y1], [x2, y2], ...].
            - int: Number of faces detected.
            - np.ndarray: A copy of the image with rectangles drawn around faces.
    """
    global face_cascade
    if face_cascade is None:
        print("Error: Face cascade classifier not loaded.")
        return [], [], 0, image.copy()

    image_copy = image.copy() # Work on a copy to draw rectangles
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=FACE_DETECT_SCALE_FACTOR,
        minNeighbors=FACE_DETECT_MIN_NEIGHBORS
    )

    face_radii = []
    face_centers = []
    face_count = 0

    for (x, y, w, h) in faces:
        radius = h / 2.0
        center_x = x + w / 2.0
        center_y = y + h / 2.0

        face_radii.append(radius)
        face_centers.append([center_x, center_y])

        # Draw a rectangle on the copy
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_count += 1

    print(f"Detected {face_count} face(s).")
    return face_radii, face_centers, face_count, image_copy

def align_images_and_create_mask(target_image, source_image, source_face_center, source_face_radius):
    """Aligns source_image to target_image using feature matching and creates a mask.

    Args:
        target_image (np.ndarray): The target image (BGR, uint8).
        source_image (np.ndarray): The source image (BGR, uint8) containing the face to warp.
        source_face_center (list[float]): The [x, y] coordinates of the center of the face in the source image.
        source_face_radius (float): The radius of the face in the source image.

    Returns:
        tuple: A tuple containing:
            - np.ndarray | None: The warped source image, aligned with the target, or None on failure.
            - np.ndarray | None: The generated mask for blending, or None on failure.
            - np.ndarray | None: The homography matrix, or None on failure.
    """
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    keypoints_target, descriptors_target = orb.detectAndCompute(target_image, None)
    keypoints_source, descriptors_source = orb.detectAndCompute(source_image, None)

    if descriptors_target is None or descriptors_source is None:
        print("Error: Could not compute descriptors for one or both images.")
        return None, None, None

    # Match descriptors using Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_source, descriptors_target) # Match source to target

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Display top matches (optional)
    match_visualization = cv2.drawMatches(
        source_image, keypoints_source,
        target_image, keypoints_target,
        matches[:FEATURE_MATCH_DISPLAY_COUNT], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(10, 5))
    plt.title('Feature Matching (Top 50)')
    plt.imshow(cv2.cvtColor(match_visualization, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for plt
    plt.axis('off')
    plt.show()

    if len(matches) < 4: # Need at least 4 matches for homography
        print("Error: Not enough matches found to compute homography.")
        return None, None, None

    # Extract locations of good matches
    source_pts = np.float32([keypoints_source[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target_pts = np.float32([keypoints_target[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find Homography using RANSAC
    homography_matrix, ransac_mask = cv2.findHomography(
        source_pts, target_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD
    )

    if homography_matrix is None:
        print("Error: Could not compute homography matrix.")
        return None, None, None

    # Warp the source image using the homography
    target_height, target_width = target_image.shape[:2]
    warped_source_image = cv2.warpPerspective(source_image, homography_matrix, (target_width, target_height))

    # --- Create the mask ---
    # Transform the source face center to the target image's coordinate system
    source_center_np = np.array([[source_face_center]], dtype=np.float32) # Shape (1, 1, 2)
    transformed_center = cv2.perspectiveTransform(source_center_np, homography_matrix)

    if transformed_center is None or transformed_center.size == 0:
        print("Error: Could not transform face center.")
        return warped_source_image, None, homography_matrix # Return warped image but no mask

    center_x = int(transformed_center[0, 0, 0])
    center_y = int(transformed_center[0, 0, 1])

    # Estimate the radius in the warped image (approximate)
    # A more accurate way would involve transforming points on the circle boundary
    # For simplicity, we scale by an average scaling factor from the homography (tricky)
    # Or just use the original radius scaled slightly.
    warped_radius = int(source_face_radius * MASK_RADIUS_MULTIPLIER) # Simple scaling

    # Create a black mask and draw a white circle
    blend_mask = np.zeros(target_image.shape[:2], dtype=np.uint8) # Grayscale mask
    cv2.circle(blend_mask, (center_x, center_y), warped_radius, (255), -1) # White filled circle

    # Optional: Apply Gaussian blur to the mask edges for smoother blending
    blend_mask = cv2.GaussianBlur(blend_mask, (5, 5), 0)

    cv2.imshow('Generated Blend Mask', blend_mask)
    # cv2.waitKey(0) # Keep commented unless interactive display is needed

    return warped_source_image, blend_mask, homography_matrix


if __name__ == '__main__':
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if face_cascade.empty():
        print(f"Error: Could not load Haar cascade from {HAAR_CASCADE_PATH}")
        sys.exit(1)
    print("Face cascade loaded successfully.")

    # Get image filenames from user
    target_filename = input('Enter the filename of the target image (where the face will be placed): ')
    source_filename = input('Enter the filename of the source image (containing the face to copy): ')

    # Load images
    target_image = cv2.imread(target_filename)
    if target_image is None:
        print(f"Error: Could not load target image: {target_filename}")
        sys.exit(1)

    source_image = cv2.imread(source_filename)
    if source_image is None:
        print(f"Error: Could not load source image: {source_filename}")
        sys.exit(1)

    print("Images loaded successfully.")

    # --- Detect faces ---
    # Detect in target image (optional, mainly for visualization/context)
    target_radii, target_centers, target_face_count, target_image_with_faces = detect_faces(target_image)

    # Detect in source image (crucial for getting the face to copy)
    source_radii, source_centers, source_face_count, source_image_with_faces = detect_faces(source_image)

    # --- Basic Error Handling for Face Detection ---
    if source_face_count == 0:
        print("Error: No faces detected in the source image. Cannot proceed.")
        cv2.imshow('Source Image (No Faces Detected)', source_image_with_faces)
        cv2.waitKey(0)
        sys.exit(1)
    elif source_face_count > 1:
        print("Warning: Multiple faces detected in the source image. Using the first detected face.")
        # Future improvement: Allow user selection or use largest face.

    # Use the first detected face from the source image
    # Note: The order from detectMultiScale isn't guaranteed, so "first" might vary.
    # A robust solution would sort by size or position if needed.
    source_face_to_use_center = source_centers[0]
    source_face_to_use_radius = source_radii[0]

    # --- Display Input Images with Detections ---
    # Resize for display
    display_width_target = int(target_image_with_faces.shape[1] * DISPLAY_RESIZE_FACTOR)
    display_height_target = int(target_image_with_faces.shape[0] * DISPLAY_RESIZE_FACTOR)
    display_target = cv2.resize(target_image_with_faces, (display_width_target, display_height_target), interpolation=cv2.INTER_AREA)

    display_width_source = int(source_image_with_faces.shape[1] * DISPLAY_RESIZE_FACTOR)
    display_height_source = int(source_image_with_faces.shape[0] * DISPLAY_RESIZE_FACTOR)
    display_source = cv2.resize(source_image_with_faces, (display_width_source, display_height_source), interpolation=cv2.INTER_AREA)

    # Concatenate for side-by-side view if possible
    # Ensure heights match for concatenation
    max_h = max(display_target.shape[0], display_source.shape[0])
    display_target_resized = cv2.resize(display_target, (int(display_target.shape[1] * max_h / display_target.shape[0]), max_h))
    display_source_resized = cv2.resize(display_source, (int(display_source.shape[1] * max_h / display_source.shape[0]), max_h))

    input_visualization = np.concatenate((display_target_resized, display_source_resized), axis=1)
    cv2.imshow('Input Images (Target | Source) with Detections', input_visualization)

    # --- Align Images and Create Mask ---
    print("Attempting to align images and create mask...")
    warped_source, blend_mask, homography = align_images_and_create_mask(
        target_image,
        source_image,
        source_face_to_use_center,
        source_face_to_use_radius
    )

    # --- Error Handling for Alignment/Masking ---
    if warped_source is None or blend_mask is None or homography is None:
        print("Error during image alignment or mask creation. Cannot proceed with blending.")
        cv2.waitKey(0) # Keep windows open
        sys.exit(1)

    print("Alignment and mask creation successful.")

    # --- Blend Images ---
    print("Performing Laplacian pyramid blending...")
    final_result = blend_images(target_image, warped_source, blend_mask)

    print("Blending complete. Displaying final result.")

    # Keep windows open until a key is pressed
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
