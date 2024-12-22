
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
    for line in lines:
        tempPoints.y_points.append(float(line))
        label = os.path.basename(path)
        name = label.split(".")
        number = name[0].split("s")
        tempPoints.labels.append(int(number[1]))
    tempPoints.x_points = list(range(0,len(tempPoints.y_points)))
    return tempPoints

def browseClick():
    # Opens the browse menu and specifies that only text files can be taken
    shared.file_var.set(filedialog.askopenfilename(title="Select a txt File", filetypes=[("Text files", "*.txt")]))


def getTestPoints():
    imported_points_1 = importFile("train/s1.txt")
    imported_points_2 = importFile("train/s2.txt")
    imported_points_3 = importFile("train/s3.txt")

    imported_points_1_2_y = np.concatenate((imported_points_1.y_points,imported_points_2.y_points),axis=0).tolist()
    imported_points_y = np.concatenate((imported_points_1_2_y,imported_points_3.y_points),axis=0).tolist()

    imported_points_1_2_L = np.concatenate((imported_points_1.labels,imported_points_2.labels),axis=0).tolist()
    imported_points_L = np.concatenate((imported_points_1_2_L,imported_points_3.labels),axis=0).tolist()

    imported_points_x = list(range(0,len(imported_points_y)))

    return points.Points(imported_points_x,imported_points_y,imported_points_L)


def preProcess(old_points = None):
    if old_points is None:
        old_points = points()
    rmv_mean_points = points.removeMean(old_points)
    b,a = points.butterworthBandpassFilter(samp_rate=1000,low_cut_off=1,high_cut_off=40)
    convolved_points = points.applybutterworthBandpassFilter(b,a,rmv_mean_points)
    normalized_points = points.normalize(convolved_points)
    resampled_points = points.downSample(normalized_points)
    segmented_points = points.segment(resampled_points)
    return segmented_points


def featureExtraction(segments_list = []):
    # Feature Extraction
    correlated_segments_list = []
    for segment in segments_list:
        correlated_segments_list.append(points.correlate(segment))

    final_segments_list = []
    for segment in correlated_segments_list:
        final_segments_list.append(points.DCT(segment))
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
