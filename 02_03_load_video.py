import cv2

cap = cv2.VideoCapture('images/shore.mov')

if not cap.isOpened():
    print('Cannot open file or video stream')

# display frames until ESC pressed
frame_count = 0
while True:
    ret, frame = cap.read()
    if ret:
        frame_count = frame_count + 1
        cv2.imshow('Frame', frame)
        # wait 25 ms for ESC key press
        if cv2.waitKey(25) & 0xFF == 27:
            break
    else:
        break

print(f'Frames captured: {frame_count}')
cap.release()
cv2.destroyAllWindows()
