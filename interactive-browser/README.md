# Interactive browser

Based on the [post](https://akshay.bio/blog/interactive-browser/). 
Contains the code in that post, which builds and displays a data browser to display the space of approved small-molecule drugs.

Instructions:
- Set up a conda environment, navigate to this folder, and run `pip install -r requirements.txt`
- Run `python preprocess_fdachems.py`. This will write a file `approved_drugs.h5` that the app uses. It will take a couple of minutes.
- Run `python scatter_chemviz.py` to deploy the app.