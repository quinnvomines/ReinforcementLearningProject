This package implements Q-LEARNING.

The package provided has all the files needed. 
To run the demonstration, please follow the steps below:

    1. Put stingray_sim package into catkin_ws/src folder

    2. Switch to catkin_ws and run "catkin_make"

    3. Source Stingray Simulation Package and Setup 
        a) "source ~/Stingray-Simulation/catkin_ws/devel/setup.bash"
        b) "source ~/Stingray-Simulation/stingray_setup.bash"

    4. I edited the wall_following_v1.launch file to run wall_follow.py so you don't have to use rosrun command.
       Instead, please run the command "roslaunch stingray_sim wall_following_v1.launch" to run the demonstration.

    5. IMPORTANT:
       When roslaunch is run, it will prompt you for input ON THE TERMINAL YOU RAN THE COMMAND.
       This can be very hard to see with all the extra output, but the message will say the following: 
       Enter 1 for learning mode. Enter 2 for best learned policy.
      
Notes:
  - Note that there may be a few videos to demonstrate that the robot meets various requirements for Q-learning.

References Used:
- Used the site below to learn how to teleport gazebo model using a service 
https://answers.gazebosim.org//question/22125/how-to-set-a-models-position-using-gazeboset_model_state-service-in-python/ 
