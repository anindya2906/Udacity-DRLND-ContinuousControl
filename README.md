##### **Udacity Deep Reinforcement Learning Nanodegree**
# Project 1: Continuous Control

![Continuous Control](images/continuous_control.gif "Continuous Control")

## **Introduction**
In this project we will train a Deep Deterministic Policy Gradient (DDPG) Agent to control a double jointed arm to move and reach target locations.

A reward of **+0.1** is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The **state space consists of 33 variables** corresponding to position, rotation, velocity, and angular velocities of the arm. Each **action is a vector with four numbers**, corresponding to torque applicable to two joints. **Every entry in the action vector should be a number between -1 and 1.**

**The environment is considered solved, when the average (over 100 episodes) score is at least +30.**

## **Getting Started**
To get started with the project, first we need to download the environment.
You can download the environment from the links given below based on your platform of choice.
- **Linux: [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)**
- **Mac OSX: [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)**
- **Windows (32-bit): [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)**
- **Windows (64-bit): [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)**

**Once the file is downloaded, please extract the zip file into the root of the project.**

### **Setup the python environment**
Please create and activate a virtual environment with conda with the following command.
```sh
conda create -n drlnd python=3.6
conda activate drlnd
```
Once the environment is active run the follwoing command from the root of the project to install the required packages.
```sh
pip install -r requirements.txt
```
Create an IPython kernel for the **drlnd** environment.
```sh
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

## **Instructions**
The Training and Testing code is implemented in the [Continuous_Control.ipynb](./Continuous_Control.ipynb) Notebook.
To start the jupyter notebook run the below command from the root of this project folder.
```sh
jupyter notebook
```
Once the jupyter notebook server is started open your browser and go to http://localhost:8888/ and click on the **Continuous_Control.ipynb** file to open the notebook. Once the Notebook is open click on **Kernel > Change Kernel > drlnd** menu to change the kernel.

Run all the cells in order to train a DDPG-Agent from scratch and test it. Once training is completed successfully the model checkpoint will be stored as **model.pt** at the root of the project folder.

**In case you dont want to train the agent from scratch then please skip the code cell which calls the train method.**
