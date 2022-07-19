import matplotlib.pyplot as plt, scipy
from IPython.display import HTML
from celluloid import Camera


def create_animation(coords, clrarr, niter=None, qnorm=True):
    """
    Params:
        coords: An array with each row representing an observation (point in scatterplot). First and second columns are treated as x and y coordinates.
        clrarr: One of two possibilities:
            - List of arrays, representing color values at each frame.
            - Function taking one argument, the iteration number. In this case niter must be specified.
    
    Returns:
        animation: A matplotlib ArtistAnimation object
    """
    fig, axes = plt.subplots(1,1, figsize=(6,6))
    plt.tight_layout()
    camera = Camera(fig)
    
    axes.axes.xaxis.set_ticks([])
    axes.axes.yaxis.set_ticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    #fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    
    num_iterations = len(clrarr) if niter is None else niter
    for it in range(num_iterations):
        if it % 10 == 0:
            print(f"iter: [{it}/{num_iterations}]")
        if niter is None:
            new_colors = clrarr[it]
        else:   # clrarr should be a function.
            # TBD: If clrarr is not a function, throw an exception
            # If clrarr is a function, assume it follows the spec above. Then:
            new_colors = clrarr(it)
        plot_arr = scipy.stats.rankdata(new_colors) if qnorm else new_colors
        axes.scatter(coords[:,0], coords[:,1], c=plot_arr, s=0.3)
        camera.snap()
        plt.show()
    animation = camera.animate()
    return animation