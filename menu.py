import tkinter as tk
import math as math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import shared as shared

bg = tk.StringVar()
fg = tk.StringVar()

bg.set("black")
fg.set("green")

def createCanvas(canvas, relwidth, relheight, relx, rely):
    width = shared.root.winfo_width()
    height = shared.root.winfo_height()
    canvas = tk.Canvas(canvas, width=width, height=height, highlightthickness=2, highlightbackground=fg.get())
    canvas.configure(bg=bg.get())
    canvas.place(relwidth=relwidth, relheight=relheight, relx=relx, rely=rely)
    shared.canvases.append(canvas)
    return canvas


def createLabel(label, canvas, highlight, relwidth, relheight, relx, rely):
    label = tk.Label(canvas, text=label, fg=fg.get(), bg=bg.get())
    shared.labels.append(label)
    if highlight == 1:
        label.configure(highlightthickness=2, highlightbackground=fg.get())
        shared.high_labels.append(label)
    label.place(relwidth=relwidth, relheight=relheight, relx=relx, rely=rely)


def createEntry(var, canvas, relwidth, relheight, relx, rely):
    entry = tk.Entry(canvas, textvariable=var)
    entry.place(relwidth=relwidth, relheight=relheight, relx=relx, rely=rely)


def createButton(label, func, canvas, relwidth, relheight, relx, rely):
    Border = tk.Frame(canvas, bd=0, highlightthickness=2, highlightcolor=fg.get(), highlightbackground=fg.get())
    Button = tk.Button(Border, text=label, command=func, width='20')
    Button.configure(fg=fg.get(), bg=bg.get(), bd=0, borderwidth=0)
    Border.place(relwidth=relwidth, relheight=relheight, relx=relx, rely=rely)
    Button.place(relwidth=1, relheight=1, relx=0, rely=0)
    shared.borders.append(Border)
    shared.buttons.append(Button)


def createGraph(points_x, points_y, graph_label, x_label, canvas):
    shownPoints_X = []
    shownPoints_Y = []
    i = 0
    if shared.startingPos_var.get() >= min(points_x) or shared.startingPos_var.get() < max(points_x):
        try:
            i = points_x.index(shared.startingPos_var.get())
        except ValueError:
            i = 0
    for x in range(40):
        try:
            shownPoints_X.append(points_x[i + x])
            shownPoints_Y.append(points_y[i + x])
        except IndexError:
            break
    
    fig, ax = plt.subplots()  # creates a figure in fig and sub-plots in ax
    # plots the graph using the original points object
    ax.stem(shownPoints_X, shownPoints_Y,linefmt='g--',markerfmt="go",basefmt="none")
    ax.axhline(y=0, color='green', linewidth=0.8)
    # if shared.startingPos_var.get() < 40 or shared.startingPos_var.get() > 40:
    #     ax.axvline(x=0, color='white', linewidth=0.8)
    ax.set_title(graph_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Amplitude")
    # creates a graph gui in original wave canvas then draws the graph and places it
    
    originalGraph = FigureCanvasTkAgg(fig, master=canvas)
    originalGraph.draw()
    originalGraph.get_tk_widget().place(relwidth=1, relheight=0.9, relx=0, rely=0.1)


def createCheck(var,onvalue,offvalue,canvas,relx,rely):
    check = tk.Checkbutton(canvas, variable=var, onvalue=onvalue, offvalue=offvalue, bg=bg.get())
    check.place(relwidth=0.05, relheight=0.1, relx=relx, rely=rely)

