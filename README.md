# License plate recognition

Python project to find polish license plates from cars
front/back images, and recognize chars on them.

*Project for Vision Systems subject on
Poznan University of Technology in Poland.*

## How it works
1. Generate template char images
   - generate images from font with Pillow
   - remove margins (cut out rectangle from contours)
   - add threshold
2. Cut out plate image from front/back car photo, based on contours detection
    - resize image
    - filter image (change color to grey, bilateral filter, gaussian blur)
    - find contours (threshold, close, dilate, erode)
    - filter contours (approx, area, aspect ratio, width, height)
    - crop plate image
    - transform cut out to rectangle
3. Cut out chars from plate image with contours detection
    - resize image
    - filter image (change color to grey, bilateral filter, gaussian blur)
    - find contours (canny, dilate, erode)
    - filter contours (area, aspect ratio, height, remove duplicates with iou, approx)
    - crop chars from image
4. Compare cut out chars with template chars
   - cv2 Template Matching with Square Difference Normed method
   - fix mistakes with probability table




