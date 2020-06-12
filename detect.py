#!/usr/bin/env python
"""
Yolo_v3 detection script.

Usage:
    python detect.py --type <video/webcam> --iou_threshold <iou threshold> --confidence_threshold <confidence threshold> --input_file <video filename>

Example:
    python detect.py --type video --iou_threshold 0.5 --confidence_threshold 0.5 --input_file data/video/sample.mp4

About:    
    __author__ = "Nithyananthan"
    __version__ = "1.0"
    __status__ = "Prototype"

Note:
    "input_file" parameter is necessary only when the "type" parameter is video
"""

import tensorflow as tf
import cv2
import argparse
import warnings
from yolo_v3 import Yolo_v3
from utils import load_class_names, draw_frame
from influxdb import InfluxDBClient
warnings.filterwarnings("ignore")

tf.compat.v1.disable_eager_execution()

_MODEL_SIZE = (416, 416)
_CLASS_NAMES_FILE = './data/labels/coco.names'
_MAX_OUTPUT_SIZE = 20

def main(type, iou_threshold, confidence_threshold, input_file):
    """Detect the no of person and chair in the meeting hall in case of video/webcam.

    Parameters
    ----------
    type : str
        Defines the type of input, video file or webcam.
    iou_threshold : float
        Max Intersection over Union that can be allowed in range of (0,1).
    confidence_threshold : float
        Likelihood that can be accepted and ranges betweeen (0,1).
    input_file : str
        Location if the input video file in case of video type.

    Returns
    -------
        None.
    """
    # Creating database in influxdb under the name detection
    client = InfluxDBClient(host="localhost",port="8086")
    client.create_database("detection")
    client.switch_database("detection")
    print("Created database with the name detection")
    
    # Loading the model with the coco dataset names
    class_names = load_class_names(_CLASS_NAMES_FILE)
    n_classes = len(class_names)
    model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE,
                    max_output_size=_MAX_OUTPUT_SIZE,
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold)
    print("Loading the model was successful.")

    if type == 'video':
        inputs = tf.compat.v1.placeholder(tf.float32, [1, *_MODEL_SIZE, 3])
        detections = model(inputs, training=False)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(scope='yolo_v3_model'))
        print("Initiating detection on sample video.")
        
        with tf.compat.v1.Session() as sess:
            # Loading the preTrained weight
            saver.restore(sess, './weights/model.ckpt')
            win_name = 'Video detection'
            cv2.namedWindow(win_name)
            cap = cv2.VideoCapture(input_file)
            frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                          cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter('./output/sample_output.mp4', fourcc, fps,
                                  (int(frame_size[0]), int(frame_size[1])))
            try:
                # Reading video to detect no of chair and person in each frames
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    resized_frame = cv2.resize(frame, dsize=_MODEL_SIZE[::-1],
                                               interpolation=cv2.INTER_NEAREST)
                    detection_result = sess.run(detections,
                                                feed_dict={inputs: [resized_frame]})
                    chair_count, person_count, ratio = draw_frame(frame, frame_size,
                                                detection_result,class_names, _MODEL_SIZE)
                    json_body=[
                                {"measurement":"meetingroom","fields":{"Chair":chair_count,
                                "People": person_count,"Ratio": ratio}
                                }
                             ]
                    client.write_points(json_body)
                    cv2.imshow(win_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    out.write(frame)
            finally:
                cv2.destroyAllWindows()
                cap.release()
                print('Detections have been saved successfully.')

    elif type == 'webcam':
        inputs = tf.compat.v1.placeholder(tf.float32, [1, *_MODEL_SIZE, 3])
        detections = model(inputs, training=False)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(scope='yolo_v3_model'))
        print("Initiating detection on sample Webcam.")

        with tf.compat.v1.Session() as sess:
            # Loading the preTrained weight
            saver.restore(sess, './weights/model.ckpt')
            win_name = 'Webcam detection'
            cv2.namedWindow(win_name)
            cap = cv2.VideoCapture(0)
            frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                          cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter('./output/sample_output.mp4', fourcc, fps,
                                  (int(frame_size[0]), int(frame_size[1])))

            try:
                # Reading video to detect no of chair and person in each frames
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    resized_frame = cv2.resize(frame, dsize=_MODEL_SIZE[::-1],
                                               interpolation=cv2.INTER_NEAREST)
                    detection_result = sess.run(detections,
                                                feed_dict={inputs: [resized_frame]})

                    chair_count, person_count, ratio = draw_frame(frame, frame_size,
                                            detection_result, class_names, _MODEL_SIZE)
                    json_body=[
                                {"measurement":"meetingroom","fields":{"Chair":chair_count,
                                "People": person_count,"Ratio": ratio}
                                }
                             ]
                    client.write_points(json_body)
                    cv2.imshow(win_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    out.write(frame)
            finally:
                cv2.destroyAllWindows()
                cap.release()
                print('Detections have been saved successfully.')

    else:
        raise ValueError("Inappropriate data type. Please choose either 'video' or 'webcam'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="video", help="video/webcam")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold for detection")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="maximum confidence allowed")
    parser.add_argument("--input_file", type=str, default="data/video/sample.mp4", help="path to video file")
    opt = parser.parse_args()
    print(opt)
    main(opt.type,opt.iou_threshold,opt.confidence_threshold,opt.input_file)

