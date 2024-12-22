
from tkinter import filedialog
from matplotlib import pyplot as plt
import numpy as np
import menu as menu
import shared as shared
import points as points
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
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
    tempPoints = points.Points()
    # Opens file using the path user gave
    file = open(path)
    lines = file.readlines()
    n = 0
    for line in lines:
        tempPoints.y_points.append(float(line))
        label = os.path.basename(path)
        name = label.split(".")
        number = name[0].split("s")
        tempPoints.labels.append(int(number[1]))
        n += 1
    tempPoints.x_points = list(range(0,len(tempPoints.y_points)))

    return tempPoints

def browseClick():
    # Opens the browse menu and specifies that only text files can be taken
    shared.file_var.set(filedialog.askopenfilename(title="Select a txt File", filetypes=[("Text files", "*.txt")]))


def getTestPoints():
    imported_points = importFile("train/s1.txt")
    importFile("train/s2.txt")
    importFile("train/s3.txt")
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
    # L_1 = 0
    # L_2 = 0
    # L_3 = 0

    return segmented_points


def featureExtraction(segments_list = []):
    # Feature Extraction
    correlated_segments_list = []
    for segment in segments_list:
        correlated_segments_list.append(points.correlate(segment.points))

    final_segments_list = []
    for segment in correlated_segments_list:
        final_segments_list.append(points.DCT(segment.points))
    return final_segments_list


def train():
    imported_points = getTestPoints()
    processed_list = preProcess(imported_points)
    final_list = featureExtraction(processed_list)

    # Classification
    X = []
    Y = []
    for segment in final_list:
        label = segment.label
        points = segment.points_y
        X.append(points)
        Y.append(label)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=44)
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train,Y_train)
    # Evaluate model
    accuracy = knn_model.score(X_test, Y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return knn_model


def test():
    return


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

    menu.createLabel("File:", main_canvas, 0, 0.2, 0.1, 0.05, 0.2)
    menu.createEntry(shared.file_var, main_canvas, 0.6, 0.1, 0.25, 0.2)

    menu.createButton("Browse", browseClick, main_canvas, 0.1, 0.1, 0.8, 0.2)
    
    menu.createButton("TEST", test, main_canvas, 0.2, 0.1, 0.4, 0.9)

    shared.root.mainloop()

train()
main()
