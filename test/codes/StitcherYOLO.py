import cv2
import numpy as np
import time

def stitch_images(left_img, right_img, final_H):
    # Get dimensions of the two images
    rows1, cols1 = right_img.shape[:2]
    rows2, cols2 = left_img.shape[:2]

    # Define corner points for both images
    points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    points2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    # Transform the corners of the left image using final_H
    points2_transformed = cv2.perspectiveTransform(points2, final_H)

    # Combine points from both images to find the bounding box
    list_of_points = np.concatenate((points1, points2_transformed), axis=0)
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel())
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel())

    # Compute translation matrix to shift the output to positive space
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    final_transformation = H_translation.dot(final_H)

    # Warp the left image using the computed transformation
    warped_left_img = cv2.warpPerspective(left_img, final_transformation, (x_max - x_min, y_max - y_min))

    # Place the right image into the output canvas
    output_img = np.zeros_like(warped_left_img)
    output_img[-y_min:rows1 - y_min, -x_min:cols1 - x_min] = right_img

    # Create a mask for the overlapping region
    overlap_mask = (warped_left_img > 0) & (output_img > 0)

    # Use addWeighted for the overlapping area
    blended_overlap = cv2.addWeighted(warped_left_img, 0.5, output_img, 0.5, 0)

    # Apply the blended overlap only where there is an overlap
    stitched_img = np.where(overlap_mask, blended_overlap, np.maximum(warped_left_img, output_img))

    return stitched_img

def stitchVideos(video1, video2, output_video, H, sample):
    height, width, _ = sample.shape
    fps = video1.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    while True:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()
        if not (ret1 and ret2):
            print("One or more videos reached the end.")
            break

        stitched_frame = stitch_images(frame1, frame2, H)

        # Optional: Crop and resize for consistency
        stitched_frame_resized = cv2.resize(stitched_frame, (width, height))
        out.write(stitched_frame_resized)

    video1.release()
    video2.release()
    out.release()
    print("Video stitching completed. Output saved to:", output_video)

def match_keypoints(key_points1, key_points2, descriptor1, descriptor2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            left_pt = key_points1[m.queryIdx].pt
            right_pt = key_points2[m.trainIdx].pt
            good_matches.append([left_pt[0], left_pt[1], right_pt[0], right_pt[1]])
    return good_matches

def main():
    start_time = time.time()  # Record start time

    video1 = cv2.VideoCapture("video1.mp4")
    video2 = cv2.VideoCapture("video2.mp4")
    output_video = "outputByYOLO.mp4"

    # Read first frames for homography estimation
    ret1, left_img = video1.read()
    ret2, right_img = video2.read()

    if not (ret1 and ret2):
        print("Error: Unable to read initial frames from videos.")
        return

    # Convert to grayscale
    l_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    r_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors using SIFT
    sift = cv2.SIFT_create()
    key_points1, descriptor1 = sift.detectAndCompute(l_img, None)
    key_points2, descriptor2 = sift.detectAndCompute(r_img, None)

    # Match keypoints
    good_matches = match_keypoints(key_points1, key_points2, descriptor1, descriptor2)
    points1 = np.float32([[pt[0], pt[1]] for pt in good_matches])
    points2 = np.float32([[pt[2], pt[3]] for pt in good_matches])

    # Compute homography
    final_H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Stitch first frames for preview
    result_img = stitch_images(left_img, right_img, final_H)

    # Display stitched image
    cv2.imshow("Stitched Preview", result_img)
    cv2.waitKey(3000)  # Show preview for 3 seconds
    cv2.destroyAllWindows()

    # Process full video stitching
    stitchVideos(video1, video2, output_video, final_H, result_img)

    end_time = time.time()  # Record end time
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
