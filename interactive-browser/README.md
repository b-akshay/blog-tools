# Interactive browser

Based on a blog post series, which builds and displays a data browser to display the space of approved small-molecule drugs:

- Part 1: [Lightweight, performant exploration of chemical space](https://akshay.bio/blog/interactive-browser/)
- Part 2: [Scalable interactive clustering](https://akshay.bio/blog/interactive-browser-part-2-clustering/)



## Instructions for local deployment

- Set up a conda environment, navigate to this folder, and run `pip install -r requirements.txt`.


- Run `python preprocess_fdachems.py`. This will write a file `approved_drugs.h5` that the app uses. It will take a couple of minutes.
- Run `python scatter_chemviz.py` to deploy the app.


## Instructions for cloud (Colab) deployment

- Run `scatter_chemviz.ipynb` in [Colab](https://colab.research.google.com/github/b-akshay/blog-tools/blob/master/interactive-browser/scatter_chemviz.ipynb). This notebook is the one which generated the [basic browser](https://akshay.bio/blog/interactive-browser/) in the first post of the series. **NOT UPDATED**: check `scatter_chemviz.py` for the latest browser.
