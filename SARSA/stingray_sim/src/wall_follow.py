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

acc_reward = 0
acc_reward_record = []
episode_record = []

#positions and orientations
start_x_pos = [-3.366, 2.254, -1.480, 1.413, 1.04, -1.57, 1.49]
start_y_pos = [-0.792, -3.490, 3.538, 0.552, 2.613, 0.491, -1.49]
start_z_orient = [0.992, 0.724, -0.700, 0.150, -0.717, -0.717, -0.682]
start_w_orient = [0.127, -0.690, -0.714, -0.989, 0.697, 0.697, 0.730]

#Q-table
Q_TABLE = {}

#Final policy
Q_FINAL_TABLE = {}
start_episode = 192
Q_FINAL_TABLE[(3, 2, 3, 1)] = [-0.32258796654053873, -0.43800248183172896, -0.5922559470349872]
Q_FINAL_TABLE[(3, 1, 1, 0)] = [-3.637246430027212, -3.626230568203939, -3.628291911902697]
Q_FINAL_TABLE[(3, 2, 3, 4)] = [-2.471212331878448, -2.459672803978421, -2.583104759021485]
Q_FINAL_TABLE[(3, 0, 3, 3)] = [-0.6492613730757457, -0.8972036328105026, -0.5949929977861637]
Q_FINAL_TABLE[(3, 3, 1, 2)] = [-0.2747737707385013, -0.8421227722584306, -1.076112984329505]
Q_FINAL_TABLE[(1, 1, 3, 4)] = [-1.3452534021448668, -1.6936884710902662, -1.6533398609787002]
Q_FINAL_TABLE[(3, 0, 1, 2)] = [-3.75502959309162, -3.7247775544984827, -3.536611375767965]
Q_FINAL_TABLE[(1, 0, 1, 2)] = [-2.251634474431774, -2.4900448899811445, -2.2355404582194742]
Q_FINAL_TABLE[(3, 3, 3, 3)] = [-2.3624864660635225, -1.0135763982768775, -1.5223652513322827]
Q_FINAL_TABLE[(3, 1, 3, 4)] = [-2.81216717880548, -2.9511220341322946, -2.8732751462295876]
Q_FINAL_TABLE[(3, 2, 1, 0)] = [-2.5768545306765485, -3.467099946526213, -3.2931628900198624]
Q_FINAL_TABLE[(1, 0, 1, 4)] = [-3.846200041994558, -2.6910019219735473, -3.8278102342510265]
Q_FINAL_TABLE[(3, 1, 1, 4)] = [-2.2699143961020654, -2.9641684236641748, -2.869802552879781]
Q_FINAL_TABLE[(3, 1, 1, 1)] = [-1.6428812427877588, -3.00011457259139, -3.0189122252672673]
Q_FINAL_TABLE[(3, 2, 3, 3)] = [-0.4151268858359107, -0.5758521145213769, -0.6171870558017316]
Q_FINAL_TABLE[(3, 0, 3, 2)] = [-1.0456144385771358, -0.8037381283136743, -0.8601425394929774]
Q_FINAL_TABLE[(3, 1, 1, 2)] = [-1.6523246651849777, -2.3392943253803358, -2.719774049958541]
Q_FINAL_TABLE[(1, 0, 3, 4)] = [-3.6172680615644195, -2.910364868829787, -3.5668575241230522]
Q_FINAL_TABLE[(3, 3, 1, 3)] = [-0.9048818329001699, -0.30660051105006897, -1.0261742079487188]
Q_FINAL_TABLE[(3, 3, 1, 0)] = [-2.128663319700773, -3.1379938260669293, -3.080082847306031]
Q_FINAL_TABLE[(3, 0, 3, 4)] = [-3.3759276454308127, -3.2902193233375763, -3.309696282729748]
Q_FINAL_TABLE[(1, 0, 1, 1)] = [-3.048003051444726, -2.512165134480958, -2.8499826804951525]
Q_FINAL_TABLE[(3, 0, 1, 1)] = [-3.043620573689094, -3.7529763822705644, -3.6028626507176438]
Q_FINAL_TABLE[(3, 3, 3, 0)] = [-1.2598757119659483, -2.264964673145931, -2.26165527546868]
Q_FINAL_TABLE[(3, 0, 1, 4)] = [-3.1019886016423994, -3.8790581217643374, -3.8806105451935395]
Q_FINAL_TABLE[(1, 3, 3, 4)] = [-2.9761353365844316, -3.5800701845460203, -3.6453448315079426]
Q_FINAL_TABLE[(1, 0, 1, 0)] = [-1.954112925108973, -1.8367581905235073, -1.7980648322619506]
Q_FINAL_TABLE[(3, 2, 1, 3)] = [-0.347753527751663, -0.8982785101525749, -1.179416230528661]
Q_FINAL_TABLE[(3, 1, 3, 2)] = [-1.6358583086837626, -1.8295472004254476, -1.035890159429532]
Q_FINAL_TABLE[(3, 2, 3, 2)] = [-0.4046485072565305, -0.8406365686851613, -0.6483618328121403]
Q_FINAL_TABLE[(3, 1, 1, 3)] = [-2.3039683227372643, -2.1960904433088353, -2.2208420974112304]
Q_FINAL_TABLE[(1, 2, 3, 4)] = [-1.013115297991789, -0.6522161129878101, -0.7190145070458853]
Q_FINAL_TABLE[(3, 3, 1, 4)] = [-1.2511810303898616, -2.9726922499458635, -2.6815981510682003]
Q_FINAL_TABLE[(3, 3, 1, 1)] = [-0.05545615670100238, -0.487499560962379, -0.6988455278079576]
Q_FINAL_TABLE[(3, 3, 3, 4)] = [-4.157592074465899, -4.775390752886494, -2.5575775075218248]
Q_FINAL_TABLE[(3, 0, 1, 0)] = [-2.2306955350460322, -3.89950924319379, -3.8820407490148847]
Q_FINAL_TABLE[(3, 3, 3, 1)] = [-0.3456325058682495, -0.4467368394864583, -0.12990081595980735]
Q_FINAL_TABLE[(3, 0, 1, 3)] = [-3.329209594206337, -3.345257543992897, -3.4471772069465643]
Q_FINAL_TABLE[(1, 0, 1, 3)] = [-1.661283121880754, -1.3087788646244245, -1.5580157516475484]
Q_FINAL_TABLE[(3, 2, 1, 2)] = [-0.36924164550539856, -0.7736587607732748, -1.064136581978215]
Q_FINAL_TABLE[(3, 3, 3, 2)] = [-1.2752951087504276, -1.3236708966877713, -1.062292460902837]
Q_FINAL_TABLE[(3, 1, 3, 3)] = [-0.2, -0.33706478105725535, -0.23071448644856785]
Q_FINAL_TABLE[(3, 2, 1, 1)] = [-0.4385176831371291, -1.0357329094883212, -1.313820111314508]
Q_FINAL_TABLE[(3, 1, 3, 1)] = [-1.5703479820907562, -1.522623205121438, -0.9984846038230394]
Q_FINAL_TABLE[(3, 2, 1, 4)] = [-1.6560900707862571, -1.6602855708246338, -1.583528103568598]


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
    global Q_TABLE, start_episode, acc_reward, acc_reward_record, episode_record

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
        episode_record.append(str(i))
        acc_reward_record.append(str(acc_reward))

        #f = open("record_rewards2.txt", "w")
        #f.write("Episodes: \n")
        #f.write(' '.join(episode_record) + "\n")
        #f.write("Accumulated Rewards: \n")
        #f.write(' '.join(acc_reward_record) + "\n")
        #f.close()

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
        
        while(not rospy.is_shutdown()):

            #Execute action
            vc = Pose2D()
            if(action == 0):
                vc.y = 0.1
                vc.theta = 0.5
            elif(action == 1):
                vc.y = 0.1
            elif(action == 2):
                vc.y = 0.1
                vc.theta = -0.5
            pub.publish(vc) #Publish velocity command
            rate.sleep()

            new_state = get_state(lidar_data) #Observe new state
            add_to_table(new_state)

            rewarded = get_reward(new_state) #Immediate reward
            acc_reward += rewarded
            print(str(print_state(new_state)) + " -> Reward: " + str(rewarded))

            next_action = 0
            next_values = Q_TABLE[new_state]
            if(random.random() > epsilon):
                #Choose action from Q values
                next_index_actions = np.where(next_values == np.max(next_values))[0] #Get available actions
                next_action = next_index_actions[random.randint(0,len(next_index_actions)-1)] #Choose action
            else:
                #Choose random action
                next_action = random.randint(0,len(next_values)-1)

            #Update Q-table
            temp_diff = (rewarded + discount_factor * Q_TABLE[new_state][next_action] - Q_TABLE[state][action])
            Q_TABLE[state][action] = Q_TABLE[state][action] + learning_rate * temp_diff

            #Go to next state and action
            state = new_state
            action = next_action
            
            #Record policy
            #f = open("policy_record2.txt", "w")
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