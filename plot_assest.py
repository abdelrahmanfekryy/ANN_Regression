import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

color = '#2288ee'
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['text.color'] = color
plt.rcParams["legend.edgecolor"] = color
plt.rcParams['figure.facecolor'] = '#22222200'
plt.rcParams['axes.edgecolor'] = color
plt.rcParams['axes.labelcolor'] = color
plt.rcParams['axes.titlecolor'] = color
plt.rcParams['xtick.color'] = color
plt.rcParams['ytick.color'] = color
plt.rcParams['legend.borderpad'] = 0.3


def Plot_Regression(model,x_train,y_train,weights_history):
    fig = plt.figure(figsize=(12,6))
    ax1 = plt.subplot2grid((16,16),(0,0), colspan=7, rowspan=13)
    ax1.scatter(x_train,y_train,s=20,c='lightgreen')
    lx = np.array([np.min(x_train),np.max(x_train)])
    ly = weights_history[0][0][0][0]*lx + weights_history[0][1][0]
    ax1.plot(lx,ly, 'b')
    ax1.set_title('Model Regression Slope')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2 = plt.subplot2grid((16,16),(0,9), colspan=7, rowspan=13)
    ax2.plot(model.history.history['loss'],ls='--')
    ax2.plot(model.history.history['val_loss'],ls='--')
    ax2.scatter(0,model.history.history['loss'][0],c='blue')
    ax2.scatter(0,model.history.history['val_loss'][0],c='orange')
    ax2.set_title('Model Loss / Epoch')
    ax2.set_ylabel('Mean Square Error')
    ax2.set_xlabel('Epoch')
    ax2.legend(['training_data', 'validation_data'], loc='upper right')

    dataSlider_ax = plt.subplot2grid((16,16),(15,0), colspan=16, rowspan=1,facecolor='lightgray')
    dataSlider = Slider(ax=dataSlider_ax,label='epoch',valmin=0,valmax=len(weights_history) - 1, valinit=0,valstep=1,visible = False)
    dataSlider.label.set_color(color)
    dataSlider_ax.set_xticks(np.linspace(0,len(weights_history)- 1,15,dtype=int))
    dataSlider_ax.xaxis.set_visible(True)
    dataSlider_ax.plot([0,0],[0,20],linewidth = 10,color='blue')

    def update(val):
        ax1.lines.clear()
        ly = weights_history[dataSlider.val][0][0][0]*lx + weights_history[dataSlider.val][1][0]
        ax1.plot(lx,ly, 'b')

        ax2.collections.clear()
        ax2.scatter(dataSlider.val,model.history.history['loss'][dataSlider.val],c='blue')
        ax2.scatter(dataSlider.val,model.history.history['val_loss'][dataSlider.val],c='orange')

        dataSlider_ax.lines.clear()
        dataSlider_ax.plot([dataSlider.val,dataSlider.val],[0,20],linewidth = 10,color='blue')

        fig.canvas.draw_idle()
            

    dataSlider.on_changed(update)

    plt.show()