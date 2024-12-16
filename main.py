
import tkinter as tk
from matplotlib import pyplot as plt
import menu as menu
import shared as shared


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


    shared.root.mainloop()

main()
