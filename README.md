# Real-Time-Human Activity Recognition from Video

This repository implements Real-Time Human Activity Recognition from Video using a pre-trained 3D convolutional ResNet-34 model to analyze activities frame by frame. Key applications include:

- Categorizing and analyzing video datasets efficiently.
- Enhancing security by detecting anomalies in surveillance footage.
- Assisting in sports analytics by tracking player movements and tactics.
- Monitoring workplace environments to ensure safety compliance.

The framework offers flexibility for customization, making it applicable across various fields for improved operational insights and efficiency.

## Data
The 3D ResNet-34 model utilized in this project is trained on the [**Kinetics dataset**](https://arxiv.org/abs/1705.06950).
#### Highlights include:
- 400 ***Real-Time-Human Activity Recognition from Video*** 
- A minimum of 400 video clips available for each class, obtained from YouTube
- Approximately 300,000 total videos
- You can view the complete list of classes that the model can identify [***here***](https://github.com/krishnapriya-nynaru/Real-Time-Human-Activity-Recognition-from-Video/blob/main/Human_Activity%20_Recognition/classes/action_recognition_kinetics.txt).

## Training
- The pre-trained weights for the 3D convolutional ResNet-34 model, which has been trained on the Kinetics dataset, can be downloaded from [***here***](https://github.com/shuvamdas/human-activity-recognition/blob/master/resnet-34_kinetics.onnx).

## Usage
1. Clone the repository: 
   ```bash
   git clone https://github.com/krishnapriya-nynaru/Real-Time-Human-Activity-Recognition-from-Video.git
2. Unzip the downloaded file: 
   ```bash
   unzip Real-Time-Human-Activity-Recognition-from-Video.zip
3. Navigate to the project directory: 
   ```bash
   cd Real-Time-Human-Activity-Recognition-from-Video
4. Install the required packages: 
   ```bash
   pip install -r requirements.txt
5. Run the activity recognition script:
   ```bash
   python main.py --model path-to-model/resnet-34_kinetics.onnx --classes path-to-classes/Class_Labels/action_recognition_kinetics.txt --input path-to-video/example_activities.mp4
## Results
Below are some results of developed model on test videos:-


![alt text](https://github.com/krishnapriya-nynaru/Real-Time-Human-Activity-Recognition-from-Video/blob/main/Human_Activity%20_Recognition/output_videos/output.gif) 
