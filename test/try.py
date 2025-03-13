import cv2
import numpy as np
import time  


def stitch_videos(video_paths, output_path):
    caps = [cv2.VideoCapture(video) for video in video_paths]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error: Failed to open video {video_paths[i]}")
            return
    frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = caps[0].get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * len(video_paths), frame_height))

    stitcher = cv2.Stitcher.create()

    while True:
        frames = []
        ret_vals = []
        for cap in caps:
            ret, frame = cap.read()
            ret_vals.append(ret)
            if ret:
                frames.append(frame)
            else:
                frames.append(None)
        if not all(ret_vals):
            print("One or more videos reached the end.")
            break
        if any(frame is None for frame in frames):
            print("Error: One or more frames are invalid!")
            break

        status, stitched_frame = stitcher.stitch(frames)

        if status == cv2.Stitcher_OK:

            #stitched_frame_cropped = stitched_frame[50:-50, 50:-50]
            stitched_frame_resized = cv2.resize(stitched_frame, (frame_width * len(video_paths), frame_height))
            out.write(stitched_frame_resized)

            if cv2.waitKey(int(1)) & 0xFF == ord('q'):
                break 
        else:
            print(f"Error during stitching. Status code: {status}")
            continue  


    for cap in caps:
        cap.release()
    out.release()
    cv2.destroyAllWindows()  


def main():
    start_time = time.time() 
    
    video_paths = ["video1.mp4", "video2.mp4"]
    output_video = "stitched_output.mp4"
    stitch_videos(video_paths, output_video)
    
    end_time = time.time() 
    elapsed_time = end_time - start_time  
    print(f"Video stitching completed!")
    print(f"Total time taken: {elapsed_time:.2f} seconds") 

if __name__ == "__main__":
    main()