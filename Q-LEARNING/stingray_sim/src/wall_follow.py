#!/usr/bin/env python

#ros imports 
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
#other imports
import numpy as np
import random
import math

#some variables and constants
learning_rate = 0.2
discount_factor = 0.8
epsilon_naught = 0.9
epsilon_dec_rate = 0.985

TOO_CLOSE = 0 
CLOSE = 1
MEDIUM = 2
FAR = 3
TOO_FAR = 4

episodes = 10000000

#positions and orientations
start_x_pos = [-3.366, 2.254, -1.480, 1.413, 1.04, -1.57, 1.49]
start_y_pos = [-0.792, -3.490, 3.538, 0.552, 2.613, 0.491, -1.49]
start_z_orient = [0.992, 0.724, -0.700, 0.150, -0.717, -0.717, -0.682]
start_w_orient = [0.127, -0.690, -0.714, -0.989, 0.697, 0.697, 0.730]

#Q-table
Q_TABLE = {}

#Final policy
Q_FINAL_TABLE = {}
start_episode = 331
Q_FINAL_TABLE[(3, 2, 3, 1)] = [-0.2351198851639371, -0.9866265966689134, -0.8308469116707032]
Q_FINAL_TABLE[(3, 1, 1, 0)] = [-3.9565758976705876, -3.981239496903914, -3.8179594415480023]
Q_FINAL_TABLE[(3, 2, 3, 4)] = [-3.1522429227713484, -3.148405380467247, -3.1082197563637237]
Q_FINAL_TABLE[(1, 1, 1, 4)] = [-0.4279666907143129, 0, -0.2560571957393846]
Q_FINAL_TABLE[(3, 0, 3, 3)] = [-0.2, -0.6203356848998429, -0.2]
Q_FINAL_TABLE[(3, 3, 1, 2)] = [-0.5920695695688486, -1.363547681008121, -1.4730444413019381]
Q_FINAL_TABLE[(1, 1, 3, 4)] = [-0.48028597869692324, -0.4449583633928912, -0.36005406720000005]
Q_FINAL_TABLE[(3, 0, 1, 2)] = [-2.3150369870295284, -3.727693077778356, -3.708389452998677]
Q_FINAL_TABLE[(3, 1, 1, 1)] = [-0.4321868349592065, -3.2839281105633256, -3.331341268882792]
Q_FINAL_TABLE[(3, 3, 3, 3)] = [-2.5932009667865423, -2.531637537903784, -1.0905075822637045]
Q_FINAL_TABLE[(3, 1, 3, 4)] = [-3.0184181903416887, -2.8470553814575275, -2.9154051307320827]
Q_FINAL_TABLE[(3, 2, 1, 0)] = [-4.041611850551001, -4.06433833237168, -3.8350936088068703]
Q_FINAL_TABLE[(1, 0, 1, 2)] = [-0.6024064403108312, -0.5714013728047858, -0.45673298851595706]
Q_FINAL_TABLE[(1, 0, 1, 4)] = [-2.2533334600299932, -2.045616317858977, -2.2623399478758888]
Q_FINAL_TABLE[(3, 1, 1, 4)] = [-1.7582056259185275, -2.229971751982133, -2.3792940075898565]
Q_FINAL_TABLE[(1, 0, 3, 4)] = [-2.201500993957855, -2.248258590512952, -1.7394580821739463]
Q_FINAL_TABLE[(3, 2, 3, 3)] = [0, 0, 0.0]
Q_FINAL_TABLE[(3, 0, 3, 2)] = [-0.5748910422874044, -0.48238242606515896, 0]
Q_FINAL_TABLE[(3, 1, 1, 2)] = [-0.6472775870032565, -1.5906718547183485, -1.8595695004464452]
Q_FINAL_TABLE[(3, 3, 1, 3)] = [-0.6625613658212868, -0.8124202877486186, -1.047720118841137]
Q_FINAL_TABLE[(3, 0, 3, 1)] = [0, 0, -0.7879488462963927]
Q_FINAL_TABLE[(3, 3, 1, 0)] = [-4.335852042822861, -4.2537383514914096, -2.7874342466547786]
Q_FINAL_TABLE[(3, 0, 3, 4)] = [-3.576705257905608, -3.314441263799508, -3.5890606897230835]
Q_FINAL_TABLE[(1, 0, 1, 1)] = [-1.859663826842641, -1.8754736298188894, -1.1649951707913777]
Q_FINAL_TABLE[(3, 0, 1, 1)] = [-1.9186599310295929, -4.233340683829017, -4.102040529231957]
Q_FINAL_TABLE[(3, 3, 3, 0)] = [-2.8109861846877253, -1.8095646317663858, -2.7963190203234594]
Q_FINAL_TABLE[(3, 0, 1, 4)] = [-3.4450005846202485, -2.0222408814761215, -3.6187172530614773]
Q_FINAL_TABLE[(1, 3, 3, 4)] = [-2.6916978185910896, -2.2393999253850683, -2.7582134224169814]
Q_FINAL_TABLE[(1, 0, 1, 0)] = [-1.0223907819357978, -1.34364608428371, -1.2507754992737514]
Q_FINAL_TABLE[(3, 2, 1, 3)] = [-0.20892063921778442, -0.7824588472329902, -0.40572211209367853]
Q_FINAL_TABLE[(3, 1, 3, 2)] = [-0.18362441514682912, -0.7057343475827117, -0.2]
Q_FINAL_TABLE[(3, 2, 3, 2)] = [-0.05201637191063996, -0.07946543169825747, -0.09277956833988832]
Q_FINAL_TABLE[(3, 1, 1, 3)] = [-0.966837440797999, -1.3109924274601732, -1.118454754259142]
Q_FINAL_TABLE[(3, 1, 3, 1)] = [-0.768375534370421, -0.5695572178730537, -0.7941138270612826]
Q_FINAL_TABLE[(3, 3, 1, 4)] = [-1.4188823701318858, -3.5336330383436265, -3.2360779289809236]
Q_FINAL_TABLE[(1, 2, 3, 4)] = [-0.3479468417139425, -0.2, -0.5303836093106338]
Q_FINAL_TABLE[(3, 3, 1, 1)] = [-0.40920247741758725, -0.764118191179616, -0.8075071322788222]
Q_FINAL_TABLE[(3, 3, 3, 4)] = [-4.293273768994325, -3.688338673977375, -1.8649070997255304]
Q_FINAL_TABLE[(3, 0, 1, 0)] = [-3.4211088048677967, -4.233941808158063, -8]
Q_FINAL_TABLE[(3, 3, 3, 1)] = [-1.566022747100321, -1.666499044141916, -0.5969263514649379]
Q_FINAL_TABLE[(3, 0, 1, 3)] = [-3.1946777930591432, -3.2128337542058727, -2.6505715435297725]
Q_FINAL_TABLE[(1, 0, 1, 3)] = [-0.7605855688568752, -0.7097073022791958, -0.7434440498239339]
Q_FINAL_TABLE[(3, 2, 1, 2)] = [-0.31894065544629085, -0.6721053273635123, -0.8720114098814207]
Q_FINAL_TABLE[(3, 3, 3, 2)] = [-2.1894425589799886, -0.3756759197375873, -2.0062576584476166]
Q_FINAL_TABLE[(3, 1, 3, 3)] = [-0.2, 0, -0.2]
Q_FINAL_TABLE[(3, 2, 1, 1)] = [-0.11780820425401381, -2.4181999548255098, -1.9813010115827894]
Q_FINAL_TABLE[(3, 2, 1, 4)] = [-1.5268875825589343, -2.0154924121034923, -2.1286142826605845]



#Add state to Q-table if not already in
def add_to_table(state):
    if(Q_TABLE.has_key(state)):
        return
    Q_TABLE[state] = [0,0,0]

#Calculate reward based on state
def get_reward(state):
    if(state[3] == TOO_CLOSE or state[3] == TOO_FAR or state[1] == TOO_CLOSE or state[0] == CLOSE):
        return -1
    return 0

#Function to print state
def print_state(state):
    states = ["", "", "", ""]
    for i in range(len(state)):
        if(state[i] == TOO_CLOSE):
            states[i] = "TOO_CLOSE"
        elif(state[i] == CLOSE):
            states[i] = "CLOSE"
        elif(state[i] == MEDIUM):
            states[i] = "MEDIUM"
        elif(state[i] == FAR):
            states[i] = "FAR"
        elif(state[i] == TOO_FAR):
            states[i] = "TOO_FAR"
    return states

#Function to get state
#Returns tuple of (left_region, front_region, right_region) with each region defined as CLOSE, MEDIUM, FAR
def get_state(lidar_data):
    #Get average of beams for each region
    left_region = np.mean(list(lidar_data[160:200])) 
    front_region = np.mean(list(lidar_data[70:110]))
    right_front_region = np.mean(list(lidar_data[30:60]))
    right_region = np.mean(list(lidar_data[0:20]) + list(lidar_data[340:360]))
    #States
    left_state = get_left_dist(left_region)
    front_state = get_front_dist(front_region)
    right_front_state = get_right_front_dist(right_front_region)
    right_state = get_right_dist(right_region)
    return (left_state, front_state, right_front_state, right_state) #Return state

#Functions to get distance, input is avg_val from beams

def get_left_dist(avg_val): #Left dist
    if(avg_val < 0.35):
        return CLOSE
    else:
        return FAR

def get_front_dist(avg_val): #Front dist
    if(avg_val < 0.6):
        return TOO_CLOSE
    elif(avg_val < 0.7):
        return CLOSE
    elif(avg_val < 0.8):
        return MEDIUM
    else:
        return FAR

def get_right_front_dist(avg_val): #Right-front dist
    if(avg_val < 0.8):
        return CLOSE
    else:
        return FAR

def get_right_dist(avg_val): #Right dist
    if(avg_val < 0.35):
        return TOO_CLOSE
    elif(avg_val < 0.7):
        return CLOSE
    elif(avg_val < 0.8):
        return MEDIUM
    elif(avg_val < 0.85):
        return FAR
    else:
        return TOO_FAR

#scan callback
lidar_data = None
def scan_callback(data):
    global lidar_data
    lidar_data = data.ranges

#Odom callback
x_pos = None
y_pos = None
def odom_callback(data):
    global x_pos, y_pos
    x_pos = data.pose.pose.position.x
    y_pos = data.pose.pose.position.y


#Main
def main():
    global Q_TABLE, start_episode

    #Initialize node
    rospy.init_node('wall_follow_node')

    #Set up topics to publish and subscribe to
    pub = rospy.Publisher('/triton_lidar/vel_cmd', Pose2D, queue_size=1000)
    rospy.Subscriber('/scan', LaserScan, scan_callback)
    rospy.Subscriber('/triton/odom', Odometry, odom_callback)
    rate = rospy.Rate(2)

    #While still running
    while(lidar_data == None and x_pos == None and y_pos == None and (not rospy.is_shutdown())):
        continue #If no lidar data, skip

    #Get input
    user_option = input("Enter 1 for learning mode. Enter 2 for best learned policy: ")
    if(user_option == 2):
        Q_TABLE = Q_FINAL_TABLE
    else:
        start_episode = 0

    #For each episode
    for i in range(start_episode, episodes):
        print("Episode: " + str(i))

        #Reset simulation
        #Teleport robot to starting position and orientation
        rospy.wait_for_service('/gazebo/set_model_state')
        ms_srv = ModelState()
        ms_srv.model_name = "triton_lidar"
        start = random.randint(0,6)
        ms_srv.pose.position.x = start_x_pos[start]
        ms_srv.pose.position.y = start_y_pos[start]
        ms_srv.pose.orientation.z = start_z_orient[start]
        ms_srv.pose.orientation.w = start_w_orient[start]
        rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)(ms_srv)

        state = get_state(lidar_data) #Get state
        add_to_table(state)

        #Do forever
        step = 0
        prev_x = []
        prev_y = []
        while(not rospy.is_shutdown()):
            values = Q_TABLE[state] #Get Q-table values

            action = 0
            epsilon = epsilon_naught * (epsilon_dec_rate** i) #Calculate epsilon
            if(random.random() > epsilon):
                #Choose action from Q values
                index_actions = np.where(values == np.max(values))[0] #Get available actions
                action = index_actions[random.randint(0,len(index_actions)-1)] #Choose action
            else:
                #Choose random action
                action = random.randint(0,len(values)-1)

            #Execute action
            vc = Pose2D()
            if(action == 0):
                vc.y = 0.1
                vc.theta = 0.25
            elif(action == 1):
                vc.y = 0.1
            elif(action == 2):
                vc.y = 0.1
                vc.theta = -0.25
            pub.publish(vc) #Publish velocity command
            rate.sleep()

            new_state = get_state(lidar_data) #Observe new state
            add_to_table(new_state)

            rewarded = get_reward(new_state) #Immediate reward
            print(str(print_state(new_state)) + " -> Reward: " + str(rewarded))

            #Update Q-table
            temp_diff = (rewarded + discount_factor * max(Q_TABLE[new_state]) - Q_TABLE[state][action])
            Q_TABLE[state][action] = Q_TABLE[state][action] + learning_rate * temp_diff
            
            print(Q_TABLE[state][action])

            #Go to next state
            state = new_state
            
            #Record policy
            #f = open("policy_record.txt", "w")
            #f.write("start_episode = " + str(i) + "\n")
            #for key in Q_TABLE:
            #    f.write("Q_FINAL_TABLE[" + str(key) + "] = " + str(Q_TABLE[key]) + "\n")
            #f.close()

            step += 1
            #Check termination
            prev_x.append(x_pos)
            prev_y.append(y_pos)
            if(len(prev_x) < 4 and len(prev_y) < 4):
                continue
            if(len(prev_x) > 4 and len(prev_y) > 4):
                prev_x.pop(0)
                prev_y.pop(0)
            if(math.sqrt((prev_x[3] - prev_x[0])**2 + (prev_y[3] - prev_y[0])**2) < 0.05):
                print("Robot got stuck")
                break
            if(step > 10000):
                print("10000 steps reached")
                break

                
            


#Entry point to program
if __name__ == "__main__":
    main()