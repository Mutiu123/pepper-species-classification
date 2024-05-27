
import numpy as np
import matplotlib.pyplot as plt

# Define data transformations for data augmentation and normalization
std = np.array([0.25, 0.25, 0.25])
mean = np.array([0.5, 0.5, 0.5])

def perfromPlot(COST,ACC):    
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(COST, color = color)
    ax1.set_xlabel('Iteration', color = color)
    ax1.set_ylabel('total loss', color = color)
    ax1.tick_params(axis = 'y', color = color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color = color)  # we already handled the x-label with ax1
    ax2.plot(ACC, color = color)
    ax2.tick_params(axis = 'y', color = color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    # Display the figure for a few seconds (e.g., 3 seconds)
    #plt.show(block=False)
    #plt.pause(3)  # Pause for 3 seconds

    # Close the figure
    #plt.close()
    


def displayData(inp, title):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    
    # Display the figure for a few seconds (e.g., 3 seconds)
    plt.show(block=False)
    plt.pause(3)  # Pause for 3 seconds

    # Close the figure
    plt.close()