#!/usr/bin/env python
"""Contains utility functions for Yolo v3 model."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from seaborn import color_palette
import cv2

def load_class_names(file_name):
    """Returns a list of class names read from `file_name`."""
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def draw_frame(frame, frame_size, boxes_dicts, class_names, model_size):
    """Draws detected boxes around chair and person alone in a video frame.
    Args:
        frame: A video frame.
        frame_size: A tuple of (frame width, frame height).
        boxes_dicts: A class-to-boxes dictionary.
        class_names: A class names list.
        model_size:The input size of the model.
    Returns:
        counter_list[1]: int
            Total number of chairs in the meeting hall.
        counter_list[0]: int
            Total number of people in the meeting hall.
        ratio: int
            Ratio of the no of person to no of chair.
    """
    boxes_dict = boxes_dicts[0]
    resize_factor = (frame_size[0] / model_size[1], frame_size[1] / model_size[0])
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    counter_chair = 1
    counter_person = 1
    prefered_class =["chair","person"]
    counter_list=[1,1]
    for cls in range(len(class_names)):
        # Draw bounding box on the detected chair and person in the frames
        if class_names[cls] in prefered_class:
            boxes = boxes_dict[cls]
            if class_names[cls]=="chair":
                counter_list[1]=len(boxes)
            else:
                counter_list[0]= len(boxes)
            color = colors[cls]
            color = tuple([int(x) for x in color])
            if np.size(boxes) != 0:
                for box in boxes:
                    xy = box[:4]
                    xy = [int(xy[i] * resize_factor[i % 2]) for i in range(4)]
                    cv2.rectangle(frame, (xy[0], xy[1]), (xy[2], xy[3]), color[::-1], 2)
                    (test_width, text_height), baseline = cv2.getTextSize(
                                                                class_names[cls],
                                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                                0.75, 1)
                    cv2.rectangle(frame, (xy[0], xy[1]),
                                (xy[0] + test_width, xy[1] - text_height - baseline),
                                color[::-1], thickness=cv2.FILLED)
                    cv2.putText(frame, class_names[cls], (xy[0], xy[1] - baseline),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
        if counter_list[1] == 0:
            # To avoid problem of creating error when there are no person in meeting hall the ratio is considered as zero.
            ratio = 0
        else:
            ratio = counter_list[0]/counter_list[1]
    return counter_list[1],counter_list[0], ratio