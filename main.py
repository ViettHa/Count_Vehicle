import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import pypyodbc as pyodbc
import time
import time
from tracker import *
import tkinter as tk

import mysql.connector

model = YOLO('best.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture('tvid.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

tracker_car = Tracker()
tracker_bus = Tracker()
tracker_truck = Tracker()
tracker_motorcycle = Tracker()

cy1 = 184
cy2 = 209
offset = 10

up_car = {}
counter_car_up = []

up_bus = {}
counter_bus_up = []

up_truck = {}
counter_truck_up = []

up_motorcycle = {}
counter_motorcycle_up = []

# Establish a connection to the MySQL Server
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="admin",
    database="test"
)

# Create a cursor object to execute SQL queries
cursor = conn.cursor()

# Define the SQL query to create a new table
create_table_query = """
CREATE TABLE IF NOT EXISTS vehicle_count (
    time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    car INT,
    motor INT,
    truck INT,
    bus INT
)
"""


# Execute the SQL query to create the table
cursor.execute(create_table_query)

# Define the SQL query to insert data into the table
insert_query = "INSERT INTO vehicle_count (car, motor, truck, bus) VALUES (%s, %s, %s, %s)"

# Define the time interval in seconds
time_interval = 30

# Initialize the time variables
start_time = time.time()
last_insert_time = start_time

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list_car = []
    list_bus = []
    list_truck = []
    list_motorcycle = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list_car.append([x1, y1, x2, y2])
        elif 'bus' in c:
            list_bus.append([x1, y1, x2, y2])
        elif 'truck' in c:
            list_truck.append([x1, y1, x2, y2])
        elif 'motorcycle' in c:
            list_motorcycle.append([x1, y1, x2, y2])

    bbox_car_idx = tracker_car.update(list_car)
    for bbox_car in bbox_car_idx:
        x, y, x2, y2, id = bbox_car
        cx = int((x + x2) / 2)
        cy = int((y + y2) / 2)

        if cy1 - offset < cy < cy1 + offset:
            up_car[id] = (cx, cy)
        if id in up_car and cy2 - offset < cy < cy2 + offset:
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 255), 2)
            cvzone.putTextRect(frame, f'{id}', (x, y), 1, 1)
            if id not in counter_car_up:
                counter_car_up.append(id)
    count_car_up = len(counter_car_up)
    cvzone.putTextRect(frame, f'upcar: {count_car_up}', (50, 50), 1, 1)

    bbox_bus_idx = tracker_bus.update(list_bus)
    for bbox_bus in bbox_bus_idx:
        x, y, x2, y2, id = bbox_bus
        cx = int((x + x2) / 2)
        cy = int((y + y2) / 2)

        if cy1 - offset < cy < cy1 + offset:
            up_bus[id] = (cx, cy)
        if id in up_bus and cy2 - offset < cy < cy2 + offset:
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 255), 2)
            cvzone.putTextRect(frame, f'{id}', (x, y), 1, 1)
            if id not in counter_bus_up:
                counter_bus_up.append(id)
    count_bus_up = len(counter_bus_up)
    cvzone.putTextRect(frame, f'upbus: {count_bus_up}', (50, 100), 1, 1)

    bbox_truck_idx = tracker_truck.update(list_truck)
    for bbox_truck in bbox_truck_idx:
        x, y, x2, y2, id = bbox_truck
        cx = int((x + x2) / 2)
        cy = int((y + y2) / 2)

        if cy1 - offset < cy < cy1 + offset:
            up_truck[id] = (cx, cy)
        if id in up_truck and cy2 - offset < cy < cy2 + offset:
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 255), 2)
            cvzone.putTextRect(frame, f'{id}', (x, y), 1, 1)
            if id not in counter_truck_up:
                counter_truck_up.append(id)
    count_truck_up = len(counter_truck_up)
    cvzone.putTextRect(frame, f'uptruck: {count_truck_up}', (50, 150), 1, 1)

    bbox_motorcycle_idx = tracker_motorcycle.update(list_motorcycle)
    for bbox_motorcycle in bbox_motorcycle_idx:
        x, y, x2, y2, id = bbox_motorcycle
        cx = int((x + x2) / 2)
        cy = int((y + y2) / 2)

        if cy1 - offset < cy < cy1 + offset:
            up_motorcycle[id] = (cx, cy)
        if id in up_motorcycle and cy2 - offset < cy < cy2 + offset:
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 255), 2)
            cvzone.putTextRect(frame, f'{id}', (x, y), 1, 1)
            if id not in counter_motorcycle_up:
                counter_motorcycle_up.append(id)
    count_motorcycle_up = len(counter_motorcycle_up)
    cvzone.putTextRect(frame, f'upmotor: {count_motorcycle_up}', (50, 200), 1, 1)

    cv2.line(frame, (1, cy1), (1018, cy1), (0, 255, 0), 2)
    cv2.line(frame, (3, cy2), (1016, cy2), (0, 0, 255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Check if the time interval has passed since the last insert
    current_time = time.time()
    if current_time - last_insert_time >= time_interval:
        # Execute the SQL query with the values of count_car_up, count_motorcycle_up, count_truck_up, count_bus_up
        cursor.execute(insert_query, (count_car_up, count_motorcycle_up, count_truck_up, count_bus_up))

        # Commit the changes to the database
        conn.commit()

        # Update the last insert time
        last_insert_time = current_time

cap.release()
cv2.destroyAllWindows()

# Close the cursor and the connection
cursor.close()
conn.close()
