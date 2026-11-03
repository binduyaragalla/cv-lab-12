# cv-lab-12
geometric transformation between two images using RANSAC algorithm

# CV Lab – Image Alignment Using RANSAC Algorithm

## 1. Aim

To estimate a robust geometric transformation between two images using the RANSAC algorithm by eliminating incorrect feature matches and computing an accurate Homography matrix for image alignment.

## 2. Software Used

- Google Colab
- Programming Language: Python
- OpenCV (cv2)
- NumPy
- Matplotlib

## 3. Theory

**RANSAC (Random Sample Consensus):**

RANSAC is an iterative algorithm used to estimate mathematical models from datasets containing outliers. It identifies the best model by selecting points that fit the model and ignoring incorrect matches.

**Main Idea:**

RANSAC finds a model that fits the majority of matching points (inliers) while ignoring incorrect points (outliers). It repeatedly selects random samples and evaluates the model until a suitable transformation is obtained.

**Homography:**

Homography is a 3×3 transformation matrix used to map points from one image to another. It helps align two images that contain common visual features.

**Steps of RANSAC Algorithm:**

1. Detect feature points in both images using ORB.
2. Extract feature descriptors from both images.
3. Match the descriptors using Brute Force Matcher.
4. Sort the matches based on their distance.
5. Select the best 15% of matches.
6. Extract the corresponding point coordinates.
7. Apply RANSAC to estimate the Homography matrix.
8. Identify inliers and eliminate outliers.
9. Apply perspective transformation to align the images.
10. Display the reference image and aligned image.

## 4. Input Images

Two input images are used in this experiment:

- **Image 1:** `images (1).jpg` – Image to be aligned.
- **Image 2:** `one-piece-wallpaper-thumb.jpg` – Reference image.

<img width="284" height="428" alt="images (1)" src="https://github.com/user-attachments/assets/e9ee15b3-5d38-47d8-8b51-8ee1e2dc595a" />


<img width="551" height="310" alt="one-piece-wallpaper-thumb" src="https://github.com/user-attachments/assets/b934ff95-188f-437b-99fe-d2389f3fe471" />


These images are used to detect and match common feature points for image alignment.

## 5. Libraries Used

- **OpenCV (`cv2`):** Used for image loading, feature detection, feature matching, Homography estimation, and image alignment.
- **NumPy (`numpy`):** Used for numerical operations and storing feature point coordinates.
- **Matplotlib (`pyplot`):** Used to visualize feature matches and display the final aligned images.

## 6. Algorithm

1. Import OpenCV, NumPy, and Matplotlib libraries.
2. Load the two input images.
3. Convert the images from BGR to RGB for display.
4. Convert both images into grayscale for feature detection.
5. Initialize the ORB feature detector with 2000 features.
6. Detect keypoints and descriptors in both images.
7. Create a Brute Force Matcher using Hamming distance.
8. Match the descriptors and sort them based on distance.
9. Select the top 15% of matches.
10. Display the feature matches before applying RANSAC.
11. Extract the coordinates of the matched keypoints.
12. Compute the Homography matrix using `cv2.findHomography()` with RANSAC.
13. Count the inliers identified by RANSAC.
14. Apply perspective transformation using `cv2.warpPerspective()`.
15. Display the reference image and the aligned image.

## 7. Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
img1 = cv2.imread('images (1).jpg')
img2 = cv2.imread('one-piece-wallpaper-thumb.jpg')

# Convert to RGB for matplotlib display
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Convert to Grayscale for feature detection
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect Features
orb = cv2.ORB_create(nfeatures=2000)

# Find keypoints and descriptors
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Match Features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(descriptors1, descriptors2)

# Sort matches based on distance
matches = sorted(matches, key=lambda x: x.distance)

# Select top 15% matches
keep_percent = 0.15
num_keep = int(len(matches) * keep_percent)
best_matches = matches[:num_keep]

# Visualization of Matches Before RANSAC
img_matches = cv2.drawMatches(
    img1_rgb, keypoints1,
    img2_rgb, keypoints2,
    best_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(15, 10))
plt.imshow(img_matches)
plt.title(f"Top {num_keep} Raw Matches (Before RANSAC)")
plt.axis('off')
plt.show()

# Extract Point Coordinates
points1 = np.zeros((len(best_matches), 2), dtype=np.float32)
points2 = np.zeros((len(best_matches), 2), dtype=np.float32)

for i, match in enumerate(best_matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Compute Homography Matrix using RANSAC
h_matrix, mask = cv2.findHomography(
    points1, points2, cv2.RANSAC, 5.0
)

# Count Inliers
matches_mask = mask.ravel().tolist()
inliers_count = np.sum(matches_mask)

print(f"Total Matches: {len(best_matches)}")
print(f"Inliers found by RANSAC: {inliers_count}")

# Apply Homography (Warp Perspective)
height, width, channels = img2.shape

aligned_img = cv2.warpPerspective(
    img1, h_matrix, (width, height)
)

# Display Final Results
plt.figure(figsize=(15, 10))

# Reference Image
plt.subplot(1, 2, 1)
plt.imshow(img2_rgb)
plt.title("Reference Image (Target)")
plt.axis('off')

# Aligned Image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
plt.title("Aligned Image (Result)")
plt.axis('off')

plt.tight_layout()
plt.show()
```

## 8. Output

The experiment produces the following results:

- **Feature Matching:** Displays the best feature matches between the two images before applying RANSAC.
- **Total Matches:** Displays the number of selected feature matches.
- **Inliers:** Displays the number of correct matches identified by RANSAC.
- **Reference Image:** Shows the target image.
- **Aligned Image:** Shows the transformed image after applying the Homography matrix.

<img width="1182" height="638" alt="cvlab-12-output" src="https://github.com/user-attachments/assets/a0e13daf-cda9-4701-8d1d-a4f4418bdda0" />


## 9. Result

The RANSAC algorithm was successfully implemented using Python and OpenCV. ORB feature detection and Brute Force Matching were used to identify common features between two images. RANSAC estimated the Homography matrix by eliminating incorrect matches and helped align the input image with the reference image.

## 10. Conclusion

Thus, image alignment using the RANSAC algorithm was successfully performed. The experiment demonstrated how feature detection, feature matching, and Homography estimation can be combined to eliminate outliers and achieve image alignment. RANSAC is useful in applications such as image stitching, panorama creation, and object recognition.
