# SelecSLS based Motion Capture System for Human Machine Interaction
## Background
This project implements a markerless motion capture system using Convolutional Neural Networks (CNNs) to estimate 3D human poses from video input—no special suits, sensors or markers required.
The system is designed for efficient, real-time pose estimation using deep learning, with applications in robotics, biomechanics, and more.

### 🔍 Key Features:
- 2D-to-3D pose estimation pipeline
- CNN-based architecture for spatial feature extraction

### 📌 Currently trained on [HumanEva](http://humaneva.is.tue.mpg.de/)

## Building and Run the Codebase
### Prerequisites
1. PyCharm IDE
2. Anaconda distribution

### Build and Run Instructions
1. Checkout the Repository.
2. Install Anaconda software.
3. Open anaconda navigator and click environments.
4. Select the `Import` button at the bottom of environments pane.
5. While `Local Drive` selected, click on the folder icon and navigate to the repository folder you just cloned.
6. select the `cudaenv.yml` file and click open
7. In `New Environment Name` section, give a name you like
8. Click `Import`.
9. Now the environment should create and let it finalize (You will need internet for package downloading).
10. After finalizing, close anaconda navigator and open repo folder in PyCharm.
11. Go to `File ➡️ Settings ➡️ Project ➡️ Python Interpreter`.
12. Then go to `Add Interpreter` on the top right and select  `Add Local Interpreter`.
13. Then click on `Select Existing` and select `Conda` as the environment type.
14. Now all the conda environments in your computer should load automatically.
15. Select the environment you previously created and you should be good to go.

### Troubleshooting Build and Run
1. Sometimes the environment might not attach to PyCharm, but you can still attach it's python interpreter and work in the environment.
2. If such issue occurs, select the environment type as `Python` rather than conda as in 13th step.
3. Now for the Python path value, select the python executable in the environment you created, your environment should typically create in `C:\Users\<Username>\anaconda3\envs\<YourEnvName>`.
4. This should add a new python interpreter and you can use it to build and run the project.
