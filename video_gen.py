import numpy as np
import cv2
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, help='direction for saving video')
    parser.add_argument('--frame_dir', type=str, help='direction for loading Image frames')
    parser.add_argument('--fps', type=int, default=20, help='fps for video generation')
    args = parser.parse_args()
    image_list = sorted(os.listdir(args.frame_dir))

    video_writer = cv2.VideoWriter(args.video_dir, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (1220, 370))
    
    for index in image_list:
        frame = cv2.imread(os.path.join(args.frame_dir, index))
        frame = cv2.resize(frame, (1220, 370))
        video_writer.write(frame)
    print('finished')
    video_writer.release()


