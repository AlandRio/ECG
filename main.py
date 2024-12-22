
from tkinter import filedialog
from matplotlib import pyplot as plt
import numpy as np
import menu as menu
import shared as shared
import points as points
import tkinter as tk
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score , classification_report,confusion_matrix
from sklearn import preprocessing
import os


def dark_mode():
    if shared.dark_mode == 0:
        shared.dark_mode = 1
        menu.bg.set("black")
        menu.fg.set("green")
        shared.dark_var.set("⋆⁺₊ 𖤓 ₊⋆")
        plt.style.use("dark_background")

    elif shared.dark_mode == 1:
        shared.dark_mode = 0
        shared.dark_var.set("⋆⁺₊ ☾⋆⁺₊⋆")
        menu.bg.set("white")
        menu.fg.set("black")
        plt.style.use('default')

    for canvas in shared.canvases:
        canvas.config(bg = menu.bg.get())
        canvas.config(highlightbackground=menu.fg.get())

    for button in shared.buttons:
        button.config(bg = menu.bg.get())
        button.config(fg = menu.fg.get())

    for label in shared.labels:
        label.config(bg = menu.bg.get())
        label.config(fg = menu.fg.get())

    for high in shared.high_labels:
        high.config(highlightbackground=menu.fg.get())

    for border in shared.borders:
        border.config(highlightcolor=menu.fg.get(), highlightbackground=menu.fg.get())


def importFile(path = ""):
    tempPoints = []
    labels = []
    # Opens file using the path user gave
    file = open(path)
    lines = file.readlines()
    for line in lines:
        tempPoints.append(float(line))
    
    label = os.path.basename(path)
    name = label.split(".")
    number = name[0].split("s")
    labels = [int(number[1])] * len(tempPoints)
    return tempPoints, labels

def browseClick():
    # Opens the browse menu and specifies that only text files can be taken
    shared.file_var.set(filedialog.askopenfilename(title="Select a txt File", filetypes=[("Text files", "*.txt")]))


def getTestPoints():
    imported_points_y_1,imported_points_L_1 = importFile("train/s1.txt")
    imported_points_y_2,imported_points_L_2 = importFile("train/s2.txt")
    imported_points_y_3,imported_points_L_3 = importFile("train/s3.txt")

    imported_points = points.Points()

    imported_points.y_points = imported_points_y_1 + imported_points_y_2 + imported_points_y_3
    imported_points.labels = imported_points_L_1 + imported_points_L_2 + imported_points_L_3
    imported_points.x_points = list(range(len(np.copy(imported_points.y_points))))
    return imported_points


def preProcess(old_points = None):
    if old_points is None:
        old_points = points()
    # Remove Mean
    print(f"Old: {len(old_points.y_points)}")
    rmv_mean_points_y = points.removeMean(np.copy(old_points.y_points))
    rmv_mean_points = points.Points()
    rmv_mean_points.y_points = np.copy(rmv_mean_points_y)
    rmv_mean_points.x_points = list(range(len(np.copy(rmv_mean_points_y))))
    rmv_mean_points.labels = np.copy(old_points.labels)
    print(f"Remove Mean: {len(rmv_mean_points.y_points)}")

    # Apply Butterworth Filter
    b,a = points.butterworthBandpassFilter(samp_rate=1000,low_cut_off=1,high_cut_off=40)
    convolved_points = points.applybutterworthBandpassFilter(b,a,rmv_mean_points)
    print(f"Butterworth: {len(convolved_points.y_points)}")

    # Normalize
    normalized_points_y = points.normalize(np.copy(convolved_points.y_points))
    normalized_points = points.Points()
    normalized_points.x_points = list(range(len(np.copy(normalized_points_y))))
    normalized_points.y_points = np.copy(normalized_points_y)
    normalized_points.labels = np.copy(convolved_points.labels)
    print(f"Normalized: {len(normalized_points.y_points)}")

    # Downsample
    resampled_points_y = points.downSample(old_points=np.copy(normalized_points.y_points))
    resampled_points_L = points.downSample(old_points=np.copy(normalized_points.labels))
    resampled_points = points.Points()
    resampled_points.y_points = np.copy(resampled_points_y)
    normalized_points.x_points = list(range(len(np.copy(resampled_points_y))))
    resampled_points.labels = np.copy(resampled_points_L)
    print(f"Resampled: {len(resampled_points.y_points)}")

    # Segment
    segmented_points = points.segment(resampled_points)
    L_1 = 0
    L_2 = 0
    L_3 = 0
    for segment in segmented_points:
        if segment.label == 1:
            L_1 += 1
        elif segment.label == 2:
            L_2 += 1
        elif segment.label == 3:
            L_3 += 1
    print(f"Segments: {len(segmented_points)}, L_1,L_2,L_3: {L_1},{L_2},{L_3}")

    return segmented_points


def featureExtraction(segments_list = None):
    if segments_list == None:
        segments_list = []

    # Feature Extraction
    print(f"old length = {len(segments_list)}")
    correlated_segments_list = []
    for segment in segments_list:
        segment_y = segment.points.y_points
        correlated_segment_y = points.correlate(segment_y)
        correlated_segment_x = list(range(len(np.copy(correlated_segment_y))))
        correlated_segment_l = segment.label

        correlated_segment_points = points.Points()
        correlated_segment_points.x_points = np.copy(correlated_segment_x)
        correlated_segment_points.y_points = np.copy(correlated_segment_y)
        correlated_segment_points.labels = [correlated_segment_l] * len(np.copy(correlated_segment_y))


        correlated_segment = points.Segment()
        correlated_segment.points = correlated_segment_points
        correlated_segment.label = correlated_segment_l

        correlated_segments_list.append(correlated_segment)

    print(f"Correlated length = {len(correlated_segments_list)}")
    
    final_segments_list = []
    for segment in correlated_segments_list:

        final_segment_y = points.DCT(segment_y)
        final_segment_x = list(range(len(np.copy(final_segment_y))))
        final_segment_l = segment.label

        final_segment_points = points.Points()
        final_segment_points.x_points = np.copy(final_segment_x)
        final_segment_points.y_points = np.copy(final_segment_y)
        final_segment_points.labels = [final_segment_l] * len(np.copy(final_segment_y))


        final_segment = points.Segment()
        final_segment.points = final_segment_points
        final_segment.label = final_segment_l

        final_segments_list.append(final_segment)
    print(f"Final length = {len(final_segments_list)}")
    return final_segments_list


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def train():
    # Get the test points (this can be customized based on the actual file paths)
    imported_points = getTestPoints()
    processed_list = preProcess(imported_points)
    final_list = featureExtraction(processed_list)

    # Prepare data for training
    X = []  # Features
    Y = []  # Labels
    for segment in final_list:
        label = segment.label
        points = segment.points.y_points
        X.append(points)
        Y.append(label)

    # Ensure X is a 2D array (KNN requires this)
    X = np.array(X)
    Y = np.array(Y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=44)

    # Initialize the KNN model with k=3
    knn_model = KNeighborsClassifier(n_neighbors=7)

    # Train the model
    knn_model.fit(X_train, Y_train)

    # Evaluate model
    Y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Return trained model for further use
    return knn_model,scaler


knn_model,scaler = train()  

def test():
    path = shared.file_var.get()
    imported_points, labels = importFile(path=path)

    # Preprocess the test data
    old_points = points.Points()
    old_points.y_points = imported_points
    old_points.x_points = list(range(len(np.copy(imported_points))))
    old_points.labels = [0] * len(np.copy(imported_points))

    processed_list = preProcess(old_points)
    final_list = featureExtraction(processed_list)

    # Prepare the test data
    X_test = []
    for segment in final_list:
        points_list = segment.points.y_points
        X_test.append(points_list)

    X_test = np.array(X_test)

    X_scaled = scaler.transform(X_test)
    # Predict using the trained KNN model
    Y_pred = knn_model.predict(X_scaled)

    # Display the prediction result
    result = Y_pred[0]
    print(f"Results: Student {result}")
    shared.test_label.set(f"Results: Student {result}")
    menu.createLabel(shared.test_label.get(), shared.root, 0, 0.2, 0.1, 0.05, 0.6)

  


def main():

    shared.root.title("ECG Based Biometrics")
    shared.root.minsize(1280, 720)
    shared.root.configure(bg=menu.bg.get())

    
    main_canvas = menu.createCanvas(shared.root,1,1,0,0)

    menu.createLabel("ECG BASED BIOMETRICS - TEAM SC_17",main_canvas,1,0.25,0.05,0.37,0)

    shared.dark_var.set("⋆⁺₊ ☾⋆⁺₊⋆")
    menu.createButton(shared.dark_var.get(), dark_mode, main_canvas, 0.125, 0.05, 0, 0)
    # Changes the style of the graph to have a dark background
    plt.style.use("dark_background")

    menu.createLabel("File:", main_canvas, 0, 0.2, 0.1, 0.05, 0.4)
    menu.createEntry(shared.file_var, main_canvas, 0.6, 0.1, 0.25, 0.4)

    menu.createButton("Browse", browseClick, main_canvas, 0.1, 0.1, 0.8, 0.4)
    shared.test_label.set("Results: ")
    menu.createLabel(shared.test_label.get(), shared.root, 0, 0.2, 0.1, 0.05, 0.6)

    menu.createButton("TEST", test, main_canvas, 0.2, 0.1, 0.4, 0.9)

    shared.root.mainloop()

main()
