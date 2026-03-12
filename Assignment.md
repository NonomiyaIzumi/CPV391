**Subject code: Computer Vision**  
**Assignment**

**Learning outcome:**  
Upon successful completion of this assignment, you will have demonstrated the abilities to:

* Use IP camera as input device and RTSP protocol to stream video to computer.  
* Processing video into image frames: input data for face detection problem.  
* Perform pre-processing steps such as light balance, noise filtering....  
* Detect face in photo (single face). This is the input data for the face recognition problem.  
* Face recognition in the input face image..

**Scenario**  
 A class at FPT University is implementing a simple attendance system. Each student will stand in front of the IP Camera, perform their attendance. The requires that the system record who is in front of the camera and the time of entering the class.

**Problem requirements**  
The attendance program has the following basic functions::  
**Function 1:** Stream video from IP camera to computer. The Real-Time Streaming Protocol (RTSP) establishes and controls either a single or several time-synchronized streams of continuous media such as audio and video. It does not typically deliver the continuous streams itself, although interleaving of the continuous media stream with the control stream is possible. In other words, RTSP acts as a “network remote control” for multimedia servers.   
**Function 2:** The video will be cropped into image frames, depending on the capture rate and resolution of the camera. Therefore, the frame cut time depends on the student's camera. The result of this step is an image containing the student's face.  
**Function 3**: Face detection: From the input image, detect the student's face position in the photo. The result of this step is a student's facial image

**Function 4:** Face recognition: From the student's face image, the system will recognize the student number, name and date and time of class.

Evaluation Criteria

| No | Criteria | Requires  | Mark | Note |
| :---- | :---- | :---- | :---- | :---- |
| 1 | Function 1: Stream video from IP camera to computer | Using RTSP to Stream video from IP camera to computer | 1 | Using mouse or keyboard |
| 2 | Function 2: cropped the video into image frames | Cropped the video into image frames | 1 | Using mouse or keyboard |
| 3 | Function 3: Face detection | Detect the student's face position in the photo. The result of this step is a student's facial image | 2 | Using mouse or keyboard |
| 4 | Function 4: Face recognion | The system will recognize the student number, name and date and time of class. | 2 | Using mouse or keyboard |
|  | Pre- processing | Perform pre-processing steps such as light balance, noise filtering | 1 | Optional |
|  | Power point file | Presenting the process of building attendance application: steps, algorithms used, data used,... | 3 | Need present and demo the application |
| 5 | Total |  | 10 |  |

