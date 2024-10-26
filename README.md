# Virtual_makeup_try_on_tool
a virtual makeup try-on tool using OpenCV, MediaPipe, Streamlit, enabling users to apply makeup effects such as lip color, blush, eyeshadow, eyeliner by uploading an image.
Extract landmarks using MediaPipe, added scaling and normalization techniques to adapt landmark coordinates.Used fillpoly,polylines to add color to image, this created unrealistic makeup effects.
To improve this, I created mask for features, used gaussianblur and addWeighted function to blend two images(image and a mask) and this helped create more realistic makeup effects.

FUNCTIONS:
detect_landmarks(src, is_stream):
OpenCV reads images in BGR format, but MediaPipe requires the input in RGB format.
cv2.cvtColor: The function converts the input image src from BGR to RGB using OpenCV cv2.cvtColor function.
The face_mesh.process() method processes the converted image to detect facial landmarks.

normalize_landmarks(landmarks, height, width, indices):
Converts the normalized landmark coordinates from MediaPipe to actual pixel coordinates based on the image's size.

lip_mask(src, points, color):
cv2.fillPoly: This OpenCV function fills a polygon defined by the points with the specified color
cv2.GaussianBlur: Smooths the edges of the mask, making the color transition softer and more natural.
The function returns the modified mask, which can then be blended with the original image to apply the lip color.

OpenCV's addWeighted function to combine two images (or an image and a mask) by applying a specified weighting to each.
output = cv2.addWeighted(src, 1.0, mask, 0.5, 0.5)
Weighted Sum Calculation: The function calculates the output pixel values using the formula:
output(x,y)=src1(x,y)×α+src2(x,y)×β+γ
Where parameters are:
α: weight for source image src1, how much of the first image is included in the output
β: weight for mask image src2, how much of the second image is included in the output
γ: brightness of image
(x,y):  represents the coordinates of the pixels in the images.

The apply_makeup function is designed to apply virtual makeup effects (like lipstick or blush) to an image.
Input Parameters:
src: The source image  
feature: A string specifying which makeup feature to apply
Ensures the input image is of type np.uint8. If not, we converts it by clipping values to the range [0, 255].

The apply_eyeliner function is designed to apply eyeliner effects to the left and right eyes in a given image based on specified landmarks.
A blank mask of the same shape as the source image is created to apply the eyeliner effect.
It retrieves the (x, y) coordinates for those indices and draws the eyeliner on the mask using cv2.polylines

The apply_eyeshadow function is designed to apply eyeshadow effects to the left and right eyes in a given image using specified landmarks.
It retrieves the (x, y) coordinates for those indices and fills the corresponding polygon area in the mask using cv2.fillPoly
