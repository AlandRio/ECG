
from tkinter import filedialog
from matplotlib import pyplot as plt
import menu as menu
import shared as shared
import points as points
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score , classification_report,confusion_matrix
from sklearn import preprocessing


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


def importFile():
    path = shared.file_var.get()
    tempPoints = points.Points()
    # Opens file using the path user gave
    file = open(path)
    lines = file.readlines()
    for line in lines:
        tempPoints.y_points.append(float(line))
    tempPoints.x_points = list(range(0,len(tempPoints.y_points)))
    return tempPoints

def browseClick():
    # Opens the browse menu and specifies that only text files can be taken
    shared.file_var.set(filedialog.askopenfilename(title="Select a txt File", filetypes=[("Text files", "*.txt")]))


def preProcess(old_points = points()):
    # Pre Processing
    rmv_mean_points = points.remove_mean(old_points)
    butterworth_coef = points.bandPassFilter(samp_freq=1000,trans_band=100,stop_atten=100,cut_off_1=1,cut_off_2=40,)
    convolved_points = points.convolve(rmv_mean_points,butterworth_coef)
    normalized_points = points.normalize(convolved_points)
    resampled_points = points.downSample(normalized_points)
    segmented_points = points.segment(resampled_points)
    return segmented_points


def featureExtraction(points_list = []):
    # Feature Extraction
    correlated_points_list = []
    for points in points_list:
        correlated_points_list.append(points.correlate(points))

    final_points_list = []
    for points in correlated_points_list:
        final_points_list.append(points.DCT(points))
    return final_points_list

def train():
    imported_points = importFile()
    processed_list = preProcess(imported_points)
    final_list = featureExtraction(processed_list)

    # Classification
    X = []
    Y = []
    for points_list in final_list:
        X.append(points_list.label)
    
    Y.append(points_list)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=44)
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train,Y_train)
    # Evaluate model
    accuracy = knn_model.score(X_test, Y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")





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
    
    menu.createButton("Train", train, main_canvas, 0.2, 0.1, 0.4, 0.9)

    shared.root.mainloop()

main()
