# AIBasedSmartDoorWithFaceMaskDetection

The main objective of the project is to develop a smart door that can accurately detect masks
over the face in public areas such as airports, railway stations, crowded markets, malls etc., 
to curtail the spread of coronavirus and thereby contributing to public healthcare.
Face mask detection is achieved using image processing to recognize whether the person is 
wearing a mask or not. Mobilenetv2 is used to extract features and classify images. The project 
is implemented using Raspberry Pi, tensor flow and OpenCV libraries of python programming 
language. The proposed technique efficiently handles the mask detection process in public 
places and if the image of the person is detected with a mask, the smart door will open 
automatically.

Variables used:
relay1 - GPIO pin of Relay 1,
relay2 - GPIO pin of Relay 2,
INIT_LR - Initial Learning Rate,
EPOCHS - Number of Iterations,
BS - Batch Size,
data - List of Images,
labels - Label of 2 categories (with_mask/without_mask)

Hardware Requirements:
Processor: Intel(R) Core(TM) i3-7020U CPU @ 2.30GHz 2.30 GHz or higher,
RAM - 8GB or higher,
RaspberryPi - 3 Model B,
SD Card - 16GB,
Battery - 4.5V,
DC Motor,
2 Channel Relay

Software Requirements:
VNC Viewer,
Raspbian OS

Compiling procedure:
1)Supply power to Raspberry Pi
2)Open VNC viewer and connect to Raspberry Pi through IP address
3)Enter the username and password for the Raspberry Pi login
4)The VNC session should start and Raspberry Pi desktop should be visible.
5)Open the terminal, and move to the Project folder: Face-Mask-Detection
6)Run the python file : python detect_mask_video.py

#python version used: 3.7.3



