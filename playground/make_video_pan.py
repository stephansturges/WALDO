from moviepy.editor import VideoFileClip, clips_array, ColorClip
import glob
import math

frame_width = 1920
frame_height = 1080

videos = [VideoFileClip(f) for f in glob.glob('output_vids/*.mp4')]

sqrt_len = math.ceil(math.sqrt(len(videos)))
tile_height = frame_height // sqrt_len
tile_width = frame_width // sqrt_len

def pad_clip(clip, target_width, target_height):
    aspect_ratio = clip.size[0] / clip.size[1]
    new_height = target_height
    new_width = int(aspect_ratio * new_height)
    resized_clip = clip.resize(height=new_height)
    pad_x = (target_width - new_width) // 2
    return resized_clip.margin(left=pad_x, right=pad_x)

padded_clips = [pad_clip(clip, tile_width, tile_height) for clip in videos]
rows = [padded_clips[i:i+sqrt_len] for i in range(0, len(padded_clips), sqrt_len)]

# Padding the last row if needed
while len(rows[-1]) < sqrt_len:
    blank_clip = ColorClip((tile_width, tile_height), color=(0, 0, 0), duration=videos[0].duration)
    rows[-1].append(blank_clip)

final_video = clips_array(rows)
final_video.write_videofile('output.mp4', fps=videos[0].fps)
