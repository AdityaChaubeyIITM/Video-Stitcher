import cv2
import numpy as np
import os
import time

def TwoInOneOut(Left, Right):
    H = None
    if len(Left.shape) != 2 or len(Right.shape) != 2:
        print("Channel Error")
        return H

    orb = cv2.ORB_create()
    kp_Left, des_Left = orb.detectAndCompute(Left, None)
    kp_Right, des_Right = orb.detectAndCompute(Right, None)

    if des_Left is None or des_Right is None:
        print("No descriptors found in one of the images.")
        return H

    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(des_Left, des_Right, k=2)

    good_matches = []
    for match in matches:
        if len(match) == 2:  # Ensure there are at least two matches
            m, n = match
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    if len(good_matches) > 20:
        LeftMatchPT = np.float32([kp_Left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        RightMatchPT = np.float32([kp_Right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(RightMatchPT, LeftMatchPT, cv2.RANSAC)

    return H

def process_videos(video_path1, video_path2, output_path):
    start_time = time.time()
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error opening video streams.")
        return

    frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height * 2))

    H = None

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        Left = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        Right = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        if H is None:
            print("Processing Homography Matrix...")
            H = TwoInOneOut(Left, Right)

        if H is not None:
            WarpImg = cv2.warpPerspective(frame2, H, (frame1.shape[1] * 2, frame1.shape[0] * 2))
            WarpImg[:frame1.shape[0], :frame1.shape[1]] = frame1
            out.write(WarpImg)

    cap1.release()
    cap2.release()
    out.release()
    print("Stitched video saved to:", output_path)

    end_time = time.time()
    print("Execution time:", round(end_time - start_time, 2), "seconds")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path1 = os.path.join(current_dir, 'video1.mp4')
    video_path2 = os.path.join(current_dir, 'video2.mp4')
    output_path = os.path.join(current_dir, 'outputByMare.mp4')

    process_videos(video_path1, video_path2, output_path)
    print("Execution completed. Stitched video is saved as 'outputByMare.mp4' in the same folder.")
