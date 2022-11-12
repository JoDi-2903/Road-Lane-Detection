import cv2 as cv

# Import haar features for car detection - credits to Kelbu (https://github.com/Kalebu/Real-time-Vehicle-Dection-Python)
cars_cascade = cv.CascadeClassifier('assets/haarcascade_car.xml')

# Detection of cars
def detect_cars(frame):
    cars = cars_cascade.detectMultiScale(frame, 1.15, 4)
    return cars

# Highlight cars -> Put rectangle and text around them
def highlight_cars(frame, cars):
    # Rectangle for every car that was found
    for (x, y, w, h) in cars:
        cv.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv.putText(frame, "Car", (x + w // 2-20, y + h + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame