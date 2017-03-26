#imported libraries
import math
from pymouse import PyMouse
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt


def box(rects, img):
    #Unpack corners of the given rectangle
    for x1, y1, x2, y2 in rects:
        #Draw the rectangle onto the frame for the given values
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img

def contours(img, currentMousePosition):
    #Create a copy of the frame given to draw onto, as contour calculation affects the frame it is performed on
    contouredFrame = np.copy(img)

    #Convert the single channel copy into a 3 channel colour image
    contouredFrame = cv2.cvtColor(contouredFrame, cv2.COLOR_GRAY2BGR)

    #Calculate contours for the given image
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #Extract largest contour
    max_area = 0
    cnt = None
    for i in range(len(contours)):
            current_cnt = contours[i]
            area = cv2.contourArea(current_cnt)
            if(area > max_area):
                max_area = area
                cnt = current_cnt
    
    #If there were was an object within the given frame to calculate contours for
    if cnt != None:
        #Draw the contours found onto the copy created at the start of the function
        cv2.drawContours(contouredFrame,[cnt],0,(0,255,255),2)

        #Calculate the convex hull for the object identified by the calculated contours
        hull = cv2.convexHull(cnt, returnPoints = False)
        
        #Calculate the image defects of the convex hull
        defects = cv2.convexityDefects(cnt,hull)

        #Create a list to hold the position of the defects after they have been filtered
        l = []

        #Iterate through the defects
        for i in range(defects.shape[0]):
            if i != 0: #First drawn isn't part of the object, so don't include
                #Unpack the current defect
                s,e,f,d = defects[i,0]

                #Further unpack values
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])

                #Even if the defect isnt drawn we still want the entire convex hull, so draw it
                cv2.line(contouredFrame,start,end,[0,255,0],2) #Convex hull

                #Signal this is the global variable chosen by the user via the trackbar
                global defect_threshold

                #Filter those that are too close to the convex hull (not indicitive of hand gesture)
                if d > defect_threshold:
                    #add the location of the defect
                    l.append(far)

                    #Draw the defect onto the frame
                    cv2.circle(contouredFrame,far,3,[0,0,255],-1) #Defects
            
        #Return the mean position of the filtered defects as well as the number of defects. (Final return is the frame this data was drawn onto)
        if(len(l) > 0):
            return l[0], len(l), contouredFrame
        else: return currentMousePosition, 0, contouredFrame
    return 0, 0, contouredFrame

def update_moving_average(currentAverage):
    global lastPosition
    global currentPosition
    if len(moving_average) < 5:
        moving_average.append(currentAverage)
    else:
        moving_average.pop(0)
        moving_average.append(currentAverage)

        #Set last position to current position
        lastPosition = currentPosition

        #And then set current position to the new mean of the moving average
        currentPosition = np.mean(moving_average, axis=0) #Gives a 2d position

def reset_moving_average():
    moving_average = []

def getJustBox(rects, img):
    #Unpack corners of the rectangle
    for x1, y1, x2, y2 in rects:

        #Derive data needed for next step.
        xRange, yRange, x, y = x2-x1,  y2-y1, x1, y1

        #Create empty list to be returned at the end of the function
        justRectangle = []

        #Iterate through values to get a 1d array of pixels instead of one in the shape of the rectangle given (e.g. a 10x10 array of pixels)
        for y2 in range(yRange):
            for x2 in range(xRange):
                justRectangle.append((img[y2+y,x2+x]))
        return np.array(justRectangle)

def getJustBoxInShape(rects, img):
    #Unpack corners of the rectangle
    for x1, y1, x2, y2 in rects:
        #Return the section of the frame that covers that rectangle
        box = img[x1:x2,y1:y2]
        return box

def overrideFrame(rects, override, img):
    for x1, y1, x2, y2 in rects:
        img[x1:x2, y1:y2] = override[:,:]
        return img

def commitAction(noDefects, currentHandPos, last_input, sprites, time_at_last_input, mouse):
    #Set the default output to "no input given"
    returnImg = sprites[0]

    #If there's an object in shot i.e. noDefects > 0
    if noDefects > 0:
        #We always want to be updating the position so that when the mouse moves a correct movement vector can be calculated
        update_moving_average(currentHandPos);
        unholdClick(mouse)
        
        #If the imput is set for "lifting the mouse off the pad" don't move the mouse and reset the movingAverage
        if(noDefects == 2):
            returnImg = sprites[4]
            #Lifting the mouse requires no time to engage
            last_input = 4

            #restart counter for input in a way that makes the progress bar fill instantly
            time_at_last_input = time.time() - 1
        #For #defects classed as "right clicking"
        elif(noDefects == 4):
            #Show that right click has been inputted
            returnImg = sprites[2]

            #Change the "last input" to right click
            last_input = 2

            #If right mouse click command has been held for long enough to be considered a genuine command
            proceed, time_at_last_input = same_input(last_input, 2)
            if(proceed):
                if(held_long_enough(time_at_last_input)):
                    rightClick(mouse)
                    moveMouse(mouse)

        #For #defects classed as "left clicking"
        elif(noDefects == 3):
            #Show that right click has been inputted
            returnImg = sprites[1]

            #Change the "last input" to right click
            last_input = 1

            #If movement left mouse click has been held for long enough to be considered a genuine command
            proceed, time_at_last_input = same_input(last_input, 1)
            if(proceed):
                if(held_long_enough(time_at_last_input)):
                    leftClick(mouse)
                    moveMouse(mouse)

        #Else we'll only move the mouse (Note this is only when mouse is not lifted i.e number of defects == 2)  
        else:
            #Show that mouse movement has been inputted
            returnImg = sprites[3]

            #Change the "last input" to mouse movement
            last_input = 3

            #If movement command has been held for long enough to be considered a genuine command
            proceed, time_at_last_input = same_input(last_input, 3)
            if(proceed):
                if(held_long_enough(time_at_last_input)):
                    moveMouse(mouse)     
        
        #If neither the input for left or right click is down, set both corresponding flags to false
        if(noDefects != 4):
            global rightClickHeld
            rightClickHeld = False

        if(noDefects != 3):
            global leftClickHeld
            leftClickHeld = False

        return returnImg, time_at_last_input, last_input
    empty, time_at_last_input = same_input(0)
    return returnImg, time_since_last, last_input

def get_progression_point(time_since_last):
    percentage = time.time() - time_since_last
    if(percentage >= 1):
        return 300
    elif(percentage > 0):
        return int(300*percentage)
    else:
        return 0

def held_long_enough(time_at_last_input):
    if(time.time() - time_at_last_input >= 1):
        return True
    else:
        return False

def same_input(last_input, givenInput):
    global time_at_last_input
    if(last_input == givenInput):
        return True, time_at_last_input
    else:
        last_input = givenInput
        return False, time.time()

def unholdClick(mouse):
    if(leftClickHeld == False):
        mouse.release(mouse.position()[0], mouse.position()[1], button=1)
    if(rightClickHeld == False):
        mouse.release(mouse.position()[0], mouse.position()[1], buton=2)

def leftClick(mouse):
    global leftClickHeld
    if(leftClickHeld == False):
        leftClickHeld = True
        mouse.press(mouse.position()[0], mouse.position()[1], button=1)

def rightClick(mouse):
    global rightClickHeld
    if(rightClickHeld == False):
        rightClickHeld = True
        mouse.press(mouse.position()[0], mouse.position()[1], button=2)

def bound_mouse_position(currentMousePos, mouse):
    #Get the screensize using windows API
    screensize = mouse.screen_size()

    #If current mouse position goes outside of that, bring it back in to the bound
    if(currentMousePos[0] < 0):
        currentMousePos[0] = 0

    if(currentMousePos[0] > screensize[0]):
        currentMousePos[0] = screensize[0]

    if(currentMousePos[1] < 0):
        currentMousePos[1] = 0

    if(currentMousePos[1] > screensize[1]):
        currentMousePos[1] = screensize[1]
    return currentMousePos

def moveMouse(mouse):
    global lastPosition
    global currentPosition
    #Make a move vector from the difference in positions 
    movementVector = [(currentPosition[0]-lastPosition[0])*2, (currentPosition[1]-lastPosition[1])*2]
    
    #And then add it to the current mouse position
    currentMousePosition = [mouse.position()[0] + movementVector[0], mouse.position()[1] + movementVector[1]]

    #Bound it so that the mouse cannot go outside the screen (it'd take a while for the mouse to move if it's position is at -10,000)
    currentMousePosition = bound_mouse_position(currentMousePosition)

    mouse.move(currentMousePosition[0], currentMousePosition[1])
    time.sleep(.01)

#These methods change threhsold values according to user input in the gui
def update_threshold(threshVal):
    global threshold
    threshold = threshVal

def update_H(threshVal):
    global channel_one_priority
    channel_one_priority = float(threshVal)/50

def update_S(threshVal):
    global channel_two_priority
    channel_two_priority = float(threshVal)/50

def update_V(threshVal):
    global channel_three_priority
    channel_three_priority = float(threshVal)/50

def update_defect_threshold(threshVal):
    global defect_threshold
    defect_threshold = threshVal

def end_program(mouse):
    #Destroy the window because the program has finished
    cv2.destroyAllWindows()

    #Relinquish controll of the camera
    cam.release()

    #Once again, unclick both mouse buttons to give user full control.
    click(mouse.position()[0], mouse.position()[1], button=2) #For right click hold and unhold
    click(mouse.position()[0], mouse.position()[1], button=1) #For left click hold and unhold

#A few booleans about user input
leftClickHeld = False
rightClickHeld = False

#Single variable to toggle debug elements
debug = True
konkaiColour = cv2.COLOR_BGR2YCrCb

#Get access to the camera and assign it to cam
cam = cv2.VideoCapture(0)

#Get some frame info, with it determine geometric properties of frame to be used later
ret, frame = cam.read()
middle = (frame.shape[0]/2, frame.shape[1]/2)
distance = (frame.shape[0]/10, frame.shape[1]/10)
rects = [[middle[0] - distance[0], middle[1] - distance[1], middle[0], middle[1]]]

#Wait for the user to press Q (when they're happy with the image)
while(True):
    # Capture frame-by-frame
    ret, frame = cam.read()
    frame = box(rects, frame)
    
    #Equalise colours Change frame to HSV then blur it to get rid of noise
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    
    #Show frame
    cv2.imshow("frame", frame)
    
    #Wait for a key to be pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Get the next frame, turn it to HSV, get the face only using detection
frame = cv2.cvtColor(frame, konkaiColour)
justRectangle = getJustBox(rects, frame)

#Now make the 
average = np.mean(justRectangle, axis =0)
averageB, averageG, averageR = average[0], average[1], average[2]

#Create a GUI for the software
cv2.namedWindow('Software')

# create trackbars for threshold changes
cv2.createTrackbar('SkinThresh','Software',1500,5000,update_threshold)
cv2.createTrackbar('H_Prio','Software',50,100,update_H)
cv2.createTrackbar('S_Prio','Software',50,100,update_S)
cv2.createTrackbar('V_Prio','Software',50,100,update_V)

#Threshold for something to be considered skin
threshold = 1500
channel_one_priority = 1
channel_two_priority = 1
channel_three_priority = 1

##Alow user to choose their threshold
while True:
    ########grab the current frame
    (grabbed, frame) = cam.read()

    #Convert it from BGR to HSV so light invariance can be applied
    frame = cv2.cvtColor(frame, konkaiColour)

    #Blur the grabbed frame for more robust isolation
    frame = cv2.GaussianBlur(frame, (25,25), 0)
    
    #Calculate the distance from the mean pixel of skin's colour to each pixel's colour
    distances = ((((frame[:,:,0]-averageB)**2)*channel_one_priority) + ((((frame[:,:,1]-averageG)**2))*channel_two_priority) + ((((frame[:,:,2]-averageR)**2))*channel_three_priority))

    #Threshold pixels too far away from mean (This results in a 1D ARRAY, conversion to 3channels is required later on)
    thresholded = np.array(np.where(distances < threshold , 255, 0), np.uint8)
    
    #Display outputs
    cv2.imshow("Software", thresholded)
    
    #Wait for a key to be pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Destroy the current window because it isn't needed anymore and user input is already recorded
cv2.destroyAllWindows()

#Recreate the window this time for defect threshold input
cv2.namedWindow('Software')

# create trackbars for threshold changes
cv2.createTrackbar('DfctThresh','Software',1500,5000,update_defect_threshold)

#Threshold for something to be considered skin
defect_threshold = 1500

#The rectangle where contour detection is applied
rects = [[0, 0, middle[1]*2, 300]]

##Allow user to chose their defect threshold
while True:
    ########grab the current frame
    (grabbed, frame) = cam.read()
    contourBox = getJustBoxInShape(rects, frame);

    #Convert it from BGR to HSV so light invariance can be applied
    contourBox = cv2.cvtColor(contourBox, konkaiColour)

    #Blur the grabbed frame for more robust isolation
    contourBox = cv2.GaussianBlur(contourBox, (25,25), 0)
    
    #Calculate the distance from the mean pixel of skin's colour to each pixel's colour
    distances = ((((contourBox[:,:,0]-averageB)**2)*channel_one_priority) + ((((contourBox[:,:,1]-averageG)**2))*channel_two_priority) + ((((contourBox[:,:,2]-averageR)**2))*channel_three_priority))

    #Threshold pixels too far away from mean (This results in a 1D ARRAY, conversion to 3channels is required later on)
    thresholded = np.array(np.where(distances < threshold , 255, 0), np.uint8)

    #Find defect information, and also draw informative infromation onto the frame
    defectMean, noDefects, contourBox = contours(thresholded, (0,0))
    
    #Show information in a frame to be analysed by user
    frame = overrideFrame(rects, contourBox, frame)

    #Display outputs
    cv2.imshow("Software", frame)
    
    #Wait for a key to be pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Destroy the current window because it isn't needed anymore and user input is already recorded
cv2.destroyAllWindows()

#Create an array to act as the data structure for the moving average
moving_average = []

#Create some variables to allow smooth mouse movement
lastPosition = [0,0]
currentPosition = [0,0]
currentMousePosition = [0,0]

#Images of all inputs expected
noInput_s = cv2.imread("sprites/default.png")
leftMouse_s = cv2.imread("sprites/LMB.png")
rightMouse_s = cv2.imread("sprites/RMB.png")
movement_s = cv2.imread("sprites/movement.png")
lifted_s = cv2.imread("sprites/lifted.png")
sprites = (noInput_s, leftMouse_s, rightMouse_s, movement_s, lifted_s)

#Input variables
time_at_last_input = time.time()
last_input = 0

#Mouse Object
mouse = PyMouse()

##Main loop
while True:
    ########grab the current frame
    (grabbed, frame) = cam.read()
    contourBox = getJustBoxInShape(rects, frame);
    ##Blur the grabbed frame for more robust isolation
    contourBox = cv2.GaussianBlur(contourBox, (5,5), 0)

    #Convert it from BGR to HSV so light invariance can be applied
    contourBox = cv2.cvtColor(contourBox, konkaiColour)

    ##Blur the grabbed frame for more robust isolation
    contourBox = cv2.GaussianBlur(contourBox, (25,25), 0)
    
    #Calculate the distance from the mean pixel of skin's colour to each pixel's colour
    distances = ((((contourBox[:,:,0]-averageB)**2)*channel_one_priority) + ((((contourBox[:,:,1]-averageG)**2))*channel_two_priority) + ((((contourBox[:,:,2]-averageR)**2))*channel_three_priority))

    #Threshold pixels too far away from mean (This results in a 1D ARRAY, conversion to 3channels is required later on)
    thresholded = np.array(np.where(distances < threshold , 255, 0), np.uint8)

    #Find defect information, and also draw informative infromation onto the frame
    defectMean, noDefects, contourBox = contours(thresholded, currentMousePosition)

    #Decide which action to perform from defect information and then perform it.
    committedAction, time_at_last_input, last_input = commitAction(noDefects, defectMean, last_input, sprites, time_at_last_input, mouse)

    #Show information in a frame to be analysed by user
    frame = overrideFrame(rects, contourBox, frame)
    
    #A frame to show how long an input has been held
    input_progress = np.zeros(shape=(10, 300, 3))

    #Draw onto the frame, a progress bar that fills as the command is held
    cv2.line(input_progress, (0,5), (get_progression_point(time_at_last_input), 5), (0,255,0), 3)

    #Display output
    cv2.imshow("Software", frame)

    #Display action being performed as a sprite and the progress to performing it
    cv2.imshow("Progression", input_progress), cv2.imshow("Action", committedAction)
    
    #Wait for a key to be pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_program(mouse)
