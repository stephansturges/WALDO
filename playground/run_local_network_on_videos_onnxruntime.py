import cv2
import numpy as np
import torch 
import onnxruntime as rt
import os
import random
import argparse
import uuid
import re

def get_resolution_from_model_path(model_path):
    # Check for rectangular pattern first
    rect_match = re.search(r"_rect_(\d+)_(\d+)_", model_path)
    if rect_match:
        return int(rect_match.group(1)), int(rect_match.group(2))
    
    # Check for the square pattern next
    square_match = re.search(r"(\d+)px", model_path)
    if square_match:
        res = int(square_match.group(1))
        return res, res  # Return height and width

    return None, None  # If neither match, return None for both dimensions

def generate_uuid():
    return str(uuid.uuid4())

def scale_based_on_bbox(bbox):
    # Compute the diagonal length of the bounding box
    diag_length = np.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)

    # Linearly scale the text size and thickness based on the diagonal length
    text_size = max(0.4, diag_length / 300)

    return text_size

def resize_and_pad(frame, expected_width, expected_height):
    if expected_width == expected_height:
        height, width, _ = frame.shape
        new_dim = min(height, width)
        start_x = (width - new_dim) // 2
        start_y = (height - new_dim) // 2
        frame = frame[start_y:start_y+new_dim, start_x:start_x+new_dim]
        
    ratio = min(expected_width / frame.shape[1], expected_height / frame.shape[0])
    new_width = int(frame.shape[1] * ratio)
    new_height = int(frame.shape[0] * ratio)
    frame = cv2.resize(frame, (new_width, new_height))
    padded_frame = np.zeros((expected_height, expected_width, 3), dtype=np.uint8)
    y_offset = (expected_height - new_height) // 2
    x_offset = (expected_width - new_width) // 2
    padded_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame

    return padded_frame


def process_frame(frame, sess, max_outputs):
    colors = {
    'car': (255, 0, 0),  # Blue
    'van': (255, 0, 0),  # Cyan
    'truck': (0, 255, 0),  # Green
    'building': (0, 42, 92),  # Brown
    'human': (203, 192, 255),  # Pink
    'gastank': (0, 255, 255),  # Yellow
    'digger': (0, 0, 255),  # Red
    'container': (255, 255, 255),  # White
    'bus': (128, 0, 128),  # Purple
    'u_pole': (255, 0, 255),  # Magenta
    'boat': (0, 0, 139),  # Dark red
    'bike': (144, 238, 144),  # Light green
    'smoke': (0, 230, 128),  # Grey
    'solarpanels': (0, 0, 0),  # Black
    'arm': (0, 0, 0),  # Black
    'plane': (255, 255, 255)  # White
    }
    names = ['car', 'van', 'truck', 'building', 'human', 'gastank', 'digger', 'container', 'bus', 'u_pole', 'boat', 'bike', 'smoke',
             'solarpanels', 'arm', 'plane']

    resolution = 960

    # Initialize a dictionary to store the count of each category (this is not really used in this vid engine)
    category_count = {name: 0 for name in names}

    image = frame.copy()
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255

    inp = {input_name: im}
    outputs = sess.run(None, inp)[0]
    thickness = 1
    category_counts = {}

    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
        box = np.array([x0, y0, x1, y1])
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score), 1)
        name = names[cls_id]
        color = colors[name]
        name += ' ' + str(score)

        text_size= scale_based_on_bbox(box)

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, thickness)
        cv2.putText(frame, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, text_size, color, thickness=thickness)
        if max_outputs is not None:
            cv2.putText(frame, f"ONNX network max Outputs: {max_outputs}", (frame.shape[1] - 250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


        # Get the name of the class without the score
        class_name = name.split()[0]

        # Increment the count for this class in the dictionary
        if class_name in category_counts:
            category_counts[class_name] += 1
        else:
            category_counts[class_name] = 1


    # Write the category counts on the frame
    y_position = 20  # Initial y position
    for category, count in category_counts.items():
        cv2.putText(frame, f"{category}: {count}", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_position += 20  # Increment the y position for the next text


    # Return the frame with detection boxes
    return frame

def save_image(output_frame, model, frame_count, UUID):
    # Create the output directory if it does not exist
    if not os.path.exists(f"./output_images/{model[:-5]}"):
        os.makedirs(f"./output_images/{model[:-5]}")

    cv2.imwrite(f"./output_images/{model[:-5]}/{UUID}_frame_{frame_count}.jpg", output_frame)



# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--frame_limit', type=int, default=None,
                    help='frame count limit (default: unlimited)')
parser.add_argument('--frame_skip', type=int, default=1,
                    help='frame skip count')
args = parser.parse_args()

for model in os.listdir("./"):
    if model.endswith(".onnx"):
        expected_width, expected_height = get_resolution_from_model_path(model)  # Get expected resolution

        max_outputs_match = re.search(r"topk.\d+", model)
        max_outputs = None
        if max_outputs_match:
            max_outputs = int(re.search(r"\d+", max_outputs_match.group(0)).group(0))



        # Initialize video writer for the output video with the expected resolution
        out = cv2.VideoWriter(f"./output_vids/{model[:-5]}.mp4", 
                               cv2.VideoWriter_fourcc(*'mp4v'), 
                               24, 
                               (expected_width, expected_height))

        print(f"Processing model: {model}")
        cuda = torch.cuda.is_available()


        # Initialize ONNX runtime session with CUDA execution
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        

        sess = rt.InferenceSession("./" + model, providers=providers)

        input_name = sess.get_inputs()[0].name

        # Iterate over video files
        for video_file in sorted(os.listdir("./input_vids")): # sort the list to maintain the same order
            if video_file.endswith(".mp4"):
                print(f"Processing video: {video_file}")

                # Initialize video capture and writer
                cap = cv2.VideoCapture("./input_vids/" + video_file)
                width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
                fps = cap.get(cv2.CAP_PROP_FPS)

                frame_count = 0
                
                uuid_of_vid_base = generate_uuid()

                # Process frames
                while cap.isOpened() and (args.frame_limit is None or frame_count < args.frame_limit):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % args.frame_skip == 0:
                        print("processing frame number : " + str(frame_count))
                        
                        ret, frame = cap.read()
                    if ret:
                        # Prepare the frame
                        frame = resize_and_pad(frame, expected_width, expected_height)
                        
                        # Process frame
                        output_frame = process_frame(frame, sess, max_outputs)
                        out.write(output_frame)
                    else:
                        break
                    frame_count += 1

                # Release capture and writer
            # Release capture and writer
            cap.release()

        out.release()

    print(f"Finished processing model: {model}")
