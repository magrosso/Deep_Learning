import numpy as np
import cv2

cap = cv2.VideoCapture('images/shore.mov')

all_rows = open('model/synset_words.txt').read().strip().split("\n")

classes = [r[r.find(' ') + 1:] for r in all_rows]

net = cv2.dnn.readNetFromCaffe('model/bvlc_googlenet.prototxt', 'model/bvlc_googlenet.caffemodel')

if not cap.isOpened():
    print('Cannot open file or video stream')

# loop through all video frames and calculate top five class predictions
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # create a blob from current frame - image size defined in synset_words.txt
    blob = cv2.dnn.blobFromImage(frame, 1, (224, 224))
    # the blob as input for inference engine
    net.setInput(blob)
    # get all probabilities from current frame
    output = net.forward()

    row = 1
    # display top 5 predictions inside frame
    for i in np.argsort(output[0])[::-1][:5]:
        frame_text = f'Probability {output[0][i] * 100:10f} % for class "{classes[i]}"'
        # frame text in blue
        cv2.putText(frame, frame_text, (0, 25 + 40 * row), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        row += 1

    cv2.imshow('Frame', frame)
    if cv2.waitKey(25) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
