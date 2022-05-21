# Interactive browser

Based on the [post](https://akshay.bio/blog/interactive-browser/). 
Contains the code in that post, which builds and displays a data browser to display the space of approved small-molecule drugs.

## Instructions for local deployment

- Set up a conda environment, navigate to this folder, and run `pip install -r requirements.txt`
- Run `python preprocess_fdachems.py`. This will write a file `approved_drugs.h5` that the app uses. It will take a couple of minutes.
- Run `python scatter_chemviz.py` to deploy the app.


## Instructions for cloud (Colab) deployment

- Run `scatter_chemviz.ipynb` in [Colab](https://colab.research.google.com/github/b-akshay/blog-tools/blob/master/interactive-browser/scatter_chemviz.ipynb). This notebook is the one which generated the [post](https://akshay.bio/blog/interactive-browser/). 