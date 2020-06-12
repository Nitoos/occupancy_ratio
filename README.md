# Occupancy Ratio of a Meeting Room.
Occupancy ratio of a meeting room is calculated by Yolo v3 algorithm that uses deep convolutional neural networks to detect Chair and People in the room and calculate its ratio. <br> <br>

## Getting started

## Prerequisites
This project is written in Python 3.7.6 using Tensorflow 2.0 (deep learning), NumPy (numerical computing), Pillow (image processing), OpenCV (computer vision) and seaborn (visualization) packages.
This project also needs anaconda, grafana, web browser, influxdb server and was tested on a windows platform.

### Anaconda Installation
For Windows:
You can refer anaconda installation on the official web page by clicking [here](https://docs.anaconda.com/anaconda/install/windows/) .

### Grafana Installation
For Windows:
1. You can refer grafana installation on the official web page by clicking [here](https://grafana.com/docs/grafana/latest/installation/windows/) .

2. Open any web browser and enter the following url with `username` and `password` as `admin` and `admin`.
```
http://localhost:3000/
```
3. On the dashboard click on `+` and click on `import`.

4. Copy paste the content from `./grafana/dashboard.json` file to the `Import via panel json` and click on `Load`


### Running InfluxDB
For Windows:
1. You can directly click [here](https://dl.influxdata.com/influxdb/releases/influxdb-1.8.0_windows_amd64.zip) for direct download.

2. Double click the influxd.exe inside the extracted folder. This influxdb server should be running on the background while testing this application.

You can refer InfluxDB installation on the official web page by clicking [here](https://portal.influxdata.com/downloads/) .

### Installing Python Prerequisites
1. Navigate to the "Anaconda Prompt (anaconda3)" in the startup menu and double click the shell command prompt.

2. Redirect to the application folder.

```
cd C:\Users\Name\path\to\the\directory
```

3. Run the .yml file with the following command line and activate the virtual environment. This will install all the python dependency including tensorflow.
```
conda env create -f environment.yml
```
```
conda activate venv
```

### Downloading official pretrained weights
For Windows:
You can download the yolov3 weights by clicking [here](https://pjreddie.com/media/files/yolov3.weights) and adding them to the weights folder.

### Save the weights in Tensorflow format
Load the weights using `load_weights.py` script. This will convert the yolov3 weights into TensorFlow .ckpt model files!
```
python load_weights.py
```

## Running the model
You can run the model using `detect.py` script. The script works on video or your webcam. Default IoU would be set if parameters are not mentioned.
### Usage
```
python detect.py --type <video/webcam> --iou_threshold <iou threshold> --confidence_threshold <confidence threshold> --input_file <video filename>
```

### Video example
You can also run the script with video files.
```
python detect.py --type video --iou_threshold 0.5 --confidence_threshold 0.5 --input_file data/video/sample.mp4
```
The detections will be saved as `sample_output.mp4` file in the output folder.

### Webcam example
The script can also be ran using your laptops webcam as the input. Example command shown below.
```
python detect.py --type webcam --iou_threshold 0.5 --confidence_threshold 0.5 
```
The detections will be saved as `sample_output.mp4` file in the output folder.

## System Cleaning
Once the execution is completed exit the virtual enviroment with the following command.
```
conda deactivate venv
```
If the Application is not going to be executed again execute the below command to remove the virtual environment.
```
conda env remove -n venv -y
```

## Note
1. Grafana and influxdb servers should be running while executing the detect.py file.

2. Press `q` anytime duing the video detection to stop the process.

## Acknowledgments
* [Yolo v3 official paper](https://arxiv.org/abs/1804.02767)