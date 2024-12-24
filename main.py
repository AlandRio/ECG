
from tkinter import filedialog
from matplotlib import pyplot as plt
import numpy as np
from scipy.fft import dct
from scipy.signal import butter, filtfilt
import menu as menu
import shared as shared
import points as points
from sklearn import metrics
from sklearn.metrics import accuracy_score , classification_report,confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.signal import resample

import os



def dark_mode():
    if shared.dark_mode == 0:
        shared.dark_mode = 1
        menu.bg.set("black")
        menu.fg.set("green")
        shared.dark_var.set("⋆⁺₊ 𖤓 ₊⋆")
        # plt.style.use("dark_background")

    elif shared.dark_mode == 1:
        shared.dark_mode = 0
        shared.dark_var.set("⋆⁺₊ ☾⋆⁺₊⋆")
        menu.bg.set("white")
        menu.fg.set("black")
        # plt.style.use('default')

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
    imported_points.y_points  = np.array(imported_points.y_points)
    imported_points.labels = imported_points_L_1 + imported_points_L_2 + imported_points_L_3
    imported_points.labels = np.array(imported_points.labels)
    imported_points.x_points = np.array(list(range(len(np.copy(imported_points.y_points)))))
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


    # Upsampling
    upsample_points_y,upsample_points_l = points.upsample(old_points=np.copy(rmv_mean_points.y_points),old_labels=np.copy(rmv_mean_points.labels))
    upsample_points = points.Points()
    upsample_points.y_points = np.array(np.copy(upsample_points_y))
    upsample_points.x_points = np.array(list(range(len(np.copy(upsample_points_y)))))
    upsample_points.labels = np.array(np.copy(upsample_points_l))

    print(f"Upsample: {len(upsample_points.y_points)},{len(upsample_points.labels)}")


    # Apply Butterworth Filter
    b,a = points.butterworthBandpassFilter(samp_rate=1000,low_cut_off=1,high_cut_off=40)
    convolved_points = points.applybutterworthBandpassFilter(b,a,upsample_points)
    print(f"Butterworth: {len(convolved_points.y_points)}")


    # Downsample
    resampled_points_y = points.downSample(old_points=np.copy(convolved_points.y_points))
    resampled_points_l = points.downSample(old_points=np.copy(convolved_points.labels))
    resampled_points = points.Points()
    resampled_points.y_points = np.array(np.copy(resampled_points_y))
    resampled_points.x_points = np.array(list(range(len(np.copy(resampled_points_y)))))
    resampled_points.labels = np.array(np.copy(resampled_points_l))
    print(f"Downsample: {len(resampled_points.y_points)},{len(resampled_points.labels)}")


    # Normalize
    normalized_points_y = points.normalize(np.copy(resampled_points.y_points))
    normalized_points = points.Points()
    normalized_points.x_points = np.array(list(range(len(np.copy(normalized_points_y)))))
    normalized_points.y_points = np.array(np.copy(normalized_points_y))
    normalized_points.labels = np.array(np.copy(resampled_points.labels))
    print(f"Normalized: {len(normalized_points.y_points)}")



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
    all_points = points.Points()
    all_points.y_points = np.copy(normalized_points.y_points)
    all_points.x_points = np.copy(normalized_points.x_points)
    all_points.labels = np.copy(normalized_points.labels)
    return segmented_points,all_points


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
        segment_y = segment.points.y_points
        final_segment_y = points.DCT(segment_y)
        # final_segment_y = dct(segment_y)
        final_segment_l = segment.label
        
        threshold = 1e-5
        non_zero_y = []
        for coef in final_segment_y:
            if np.abs(coef) > threshold:
                non_zero_y.append(coef)
            else:
                non_zero_y.append(0)
        final_segment_points = points.Points()
        final_segment_points.y_points = np.array(np.copy(non_zero_y))
        final_segment_points.x_points = np.array(list(range(len(np.copy(non_zero_y)))))
        final_segment_points.labels = [final_segment_l] * len(np.copy(non_zero_y))


        final_segment = points.Segment()
        final_segment.points = final_segment_points
        final_segment.label = final_segment_l

        final_segments_list.append(final_segment)
    
    # Plot the results
    # plt.figure(figsize=(10, 6))
    # num1 = 20
    # num2 = 40
    # num3 = 80
    # plt.plot(final_segments_list[num1].points.y_points[0:50], label=final_segments_list[num1].label, linestyle='--')
    # plt.plot(final_segments_list[num2].points.y_points[0:50], label=final_segments_list[num2].label, linestyle='--')
    # plt.plot(final_segments_list[num3].points.y_points[0:50], label=final_segments_list[num3].label, linestyle='--')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Amplitude')
    # plt.title('Butterworth Bandpass Filter (1-40 Hz)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    return final_segments_list




def train():
    # Get the test points (this can be customized based on the actual file paths)
    imported_points = getTestPoints()
    processed_list,all_points = preProcess(imported_points)
    final_list = featureExtraction(processed_list)

    # Prepare data for training
    X = []  # Features
    Y = []  # Labels
    for segment in final_list:
        label = segment.label
        points_list = np.array(segment.points.y_points)
        X.append(points_list)
        Y.append(label)

    # Ensure X is a 2D array (KNN requires this)
    X = np.array(X)
    Y = np.array(Y)


    print(f"{X}")
    print(f"{Y}")
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=300)

    param_grid = {
    'n_neighbors': range(1, 50)
    }
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, Y_train)
    print(f"Best k: {grid_search.best_params_}")
    # Initialize the KNN model with k=3
    knn_model = KNeighborsClassifier(n_neighbors=3)

    # Train the model
    knn_model.fit(X_train, Y_train)

    scores = cross_val_score(knn_model, X, Y, cv=5)
    print(f"Cross-validation accuracy: {scores.mean()}")
    # Evaluate model
    Y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Return trained model for further use
    return knn_model


knn_model = train()  

def test():
    path = shared.file_var.get()
    imported_points, labels = importFile(path=path)

    # Preprocess the test data
    old_points = points.Points()
    old_points.y_points = imported_points
    old_points.x_points = list(range(len(np.copy(imported_points))))
    old_points.labels = labels

    processed_list,all_points = preProcess(old_points)
    final_list = featureExtraction(processed_list)

    # Prepare the test data
    X_test = []
    for segment in final_list:
        points_list = segment.points.y_points
        X_test.append(points_list)

    X_test = np.array(X_test)

    # Predict using the trained KNN model
    Y_pred = knn_model.predict(X_test)%3
    result = Y_pred[0]
    # Display the prediction result
    print(f"Results: Student {result  + 1}")
    shared.test_label.set(f"Results: Student {result  + 1}")
    menu.createLabel(shared.test_label.get(), shared.root, 0, 0.6, 0.1, 0.2, 0.1,font_size=60)
    menu.createGraph(points_x=all_points.x_points[0:800],points_y=all_points.y_points[0:800],pos=0.05,graph_label=f"Student {result + 1} Beat",x_label="Time",canvas=shared.root,count=800)
    menu.createGraph(points_x=all_points.x_points[0:800],points_y=X_test[0],pos=0.5,graph_label=f"Student {result + 1} Coeff",x_label="Time",canvas=shared.root,count=50)

  


def main():

    shared.root.title("ECG Based Biometrics")
    shared.root.minsize(1280, 720)
    shared.root.configure(bg=menu.bg.get())

    
    main_canvas = menu.createCanvas(shared.root,1,1,0,0)

    menu.createLabel("ECG BASED BIOMETRICS - TEAM SC_17",main_canvas,1,0.5,0.05,0.25,0)

    shared.dark_var.set("⋆⁺₊ ☾⋆⁺₊⋆")
    menu.createButton(shared.dark_var.get(), dark_mode, main_canvas, 0.125, 0.05, 0, 0)
    # Changes the style of the graph to have a dark background
    # plt.style.use("dark_background")

    menu.createLabel("File:", main_canvas, 0, 0.2, 0.1, 0.05, 0.2)
    menu.createEntry(shared.file_var, main_canvas, 0.6, 0.1, 0.25, 0.2)

    menu.createButton("Browse", browseClick, main_canvas, 0.1, 0.1, 0.8, 0.2)
    shared.test_label.set("Results: ")
    menu.createLabel(shared.test_label.get(), shared.root, 0, 0.6, 0.1, 0.2, 0.1,font_size=60)

    menu.createButton("TEST", test, main_canvas, 0.2, 0.1, 0.4, 0.9)

    shared.root.mainloop()

main()
