# Import required modules
from imutils.object_detection import non_max_suppression 
import cv2 as cv
import math
import argparse
import urllib.request
import numpy as np
import pytesseract

# The url

url = 'http://192.168.1.144:8080/shot.jpg'

parser = argparse.ArgumentParser(description='Use this script to run text detection deep learning networks using OpenCV.')
# Input argument
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
# Model argument
parser.add_argument('--model', default="frozen_east_text_detection.pb",
                    help='Path to a binary .pb file of model contains trained weights.'
                    )
# Width argument
parser.add_argument('--width', type=int, default=320,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.'
                   )
# Height argument
parser.add_argument('--height',type=int, default=320,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.'
                   )
# Confidence threshold
parser.add_argument('--thr',type=float, default=0.5,
                    help='Confidence threshold.'
                   )
# Non-maximum suppression threshold
parser.add_argument('--nms',type=float, default=0.4,
                    help='Non-maximum suppression threshold.'
                   )

parser.add_argument("--p", "--padding", type=float, default=0.0,
        help="amount of padding to add to each border of ROI")



args = parser.parse_args()


############ Utility functions ############
def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
           # offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
           # p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
           # p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
           # center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
           # detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
           # confidences.append(float(score))
            endX = int(offsetX + (cosA * x1_data[x])+ (sinA * x2_data[x]) )
            endY = int(offsetX - (sinA  * x1_data[x ]) + (cosA*x2_data[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            detections.append((startX, startY, endX, endY))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

if __name__ == "__main__":
    # Read and store arguments
    confThreshold = args.thr
    nmsThreshold = args.nms
    inpWidth = args.width
    inpHeight = args.height
    model = args.model
    padding = args.p

    
    # Load network
    net = cv.dnn.readNet(model)

    #Convert the image to Opencv Image

   # opencvImage = cv.imdecode(numpyImage,-1)

    # Create a new named window
    kWinName = "EAST: An Efficient and Accurate Scene Text Detector"
    cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")

   # imageResult = urllib.request.urlopen(url)


    # Open a video file or an image file or a camera stream
   # cap = cv.VideoCapture(url)

    while cv.waitKey(1) < 0:
         cap = cv.VideoCapture(url)

         # read Frame
         hasFrame, frame = cap.read()
         if not hasFrame or cv.waitKey(4)==ord("q") :
            cv.waitKey()
            break

         # image =np.array(bytearray(frame), dtype=np.uint8)
         # image = cv.imdecode(image,cv.IMREAD_COLOR)

         orig = frame.copy()
         # Get frame height and width
         height_ = frame.shape[0]
         width_ = frame.shape[1]
         rW = width_ / float(inpWidth)
         rH = height_ / float(inpHeight)

         # Create a 4D blob from frame.
         blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

         # Run the model
         net.setInput(blob)
         output = net.forward(outputLayers)
         t, _ = net.getPerfProfile()
         label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

         # Get scores and geometry
         scores = output[0]
         geometry = output[1]
         #decode the predictions, then apply the non-maxima suppression to
         #suppress weak, overlapping bounding boxes
         [boxes, confidences] = decode(scores, geometry, confThreshold)
         boxesRects = non_max_suppression(np.array(boxes), probs=confidences)
      # Apply NMS
        # indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
         #Initialize the list of results
         results = []
         for (startX, startY, endX, endY) in boxesRects:
            #Scale the bounding box coordinates based on the respective
            #ratios

            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            #in order to obtain a better OCR of the text we can potentially
            #apply a bit of padding surrounding the bounding box -- here we
            #computing the deltas in both the x and y directions

            dX = int((endX - startX) * padding)
            dY = int((endY - startY) * padding)

            #apply padding to each side of the bounding box, respectively

            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(width_, endX +(dX*2))
            endY = min(height_, endY + (dY*2))

            # extract the actual padded ROI

            roi = orig[startY:endY, startX:endX]

            # in order to apply Tesseract v4 to OCR text we must supply
            # (1) a language, (2) an OEM flag of 4, indicating that the we
            # wish to use the LSTM neural net model for OCR, and finally
            # (3) an OEM value, in this case, 7 which implies that we are
            # treating the ROI as a single line of text

            config = ("-l eng --oem 1 --psm 7")
            text = pytesseract.image_to_string(roi, config = config)


            #add the bounding box coordinates and OCR'd text to the list
            #of results

            results.append(((startX, startY, endX, endY), text ))


         results = sorted(results, key=lambda r:r[0][1])

         for((startX, startY, endX, endY), text) in results:


             print("OCR TEXT")
             print("=======")
             print("{}\n".format(text))

             #strip out non-ASCII text so we can draw the text on the image
             #using OpenCV, then draw the text and bounding box surrounding 
             #the text region of the input image

             text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
             output = orig.copy()
             cv.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)
             cv.putText(frame, text, (startX,startY-20), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255),3)
             cv.putText(frame, label, (0,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

             # Display the frame
             cv.imshow(kWinName, frame)
           # cv.imwrite(file, frame)
            # cv.imwrite("out-{}.jpg",frame)
