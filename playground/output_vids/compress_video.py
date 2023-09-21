from moviepy.editor import VideoFileClip

def compress_video(video_file, output_file, bitrate="10000k"):
    clip = VideoFileClip(video_file)
    # Compress video
    clip.write_videofile(output_file, bitrate=bitrate)

if __name__ == "__main__":
    compress_video("yolov7-W25_rect_1920_1088_newDefaults-bs48-last-topk_9999.mp4", "compressed_video2.mp4")

