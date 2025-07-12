#  Final report of research project

##  Implementation Process

The research project is implemented in a fully **modular structure**, which separates the project logic from utility functions.  
This architecture improves **readability**, **code organization**, and ensures that the system remains **maintainable and extensible** as more features are added.

###  Codebase Structure

The codebase is divided into two main components: `utils/` and `logic/`.


### `utils/` Directory:

- **`config.py`**  
  Manages all project configurations using a single `@dataclass`.  
  This approach provides a robust and easy-to-maintain way to manage parameters across the codebase.

- **`visualization.py`**  
  Contains functions for visualizing model accuracies.  
  This simplifies comparing performances between models and generating final evaluation reports.


### `logic/` Directory:

- **`main.py`**  
  The main entry point of the project that handles the entire pipeline.

- **`dataset.py`**  
  Implements `MNISTDataSet`, inheriting from `torch.utils.data.Dataset`.  
  It handles downloading, storing, and transforming the dataset.  
  *(See the Challenges section for more details.)*

- **`data_loader.py`**  
  Separates the data loading pipeline from the main logic to ensure modularity and clarity.

- **`zero_shot.py`**  
  Implements the **Zero-Shot classification model**.

- **`linear_probe.py`**  
  Implements the **Linear Probe evaluation**.  
  It extracts image features using the pre-trained OpenAI CLIP model and trains a `LogisticRegression` classifier (as mentioned by the Openai/Clip documention)
---

## Challenges Faced

### Challenge 1 : 

- **Problem**  
   The default preprocessing function(its acctualy has transforms.Compose type) returned by **clip.load()** is designed for **single image**, not for batches. Applying this transformation to each image for the entire dataset was extremely slow — **about 40 minutes on CUDA**.

- **Solution**  
     to resolve this problem, Re implemented the CLIP transformation
     pipeline inside the custom `MNISTDataSet` class. This new
     implementation preprocesses all images as tensors in **batches before
     evaluation**, allowing much larger batch sizes and enabling full GPU 
     acceleration.
     The improved pipeline **reduced evaluation time from 40 minutes → 5 minutes**.
     *(Timing comparisons are shown in accompanying visual reports.)*

### Challenge 2 : 

- **Problem**  
  as the research description mentined, **class name variants** were required for zero-shot evaluation. The first version of code (naive approach) used to run **full evaluation for just one class**, resulting in resulting in redundant computation and performance drop.

- **Solution** 
  Implemented a pipeline to handle **all class name variations in a single evaluation**.
  - the text for all variations of class names are **tokenized and encoded once** in models initialization.
  - during evaluation, the model **iterates over all class variants** for each batch and makes predictions.
   this implemention also avoids another problem (code details available in git history), which is recomputation of text features so its **redundant computation and leed us to boosts model performance**.

---

## Potential Improvements

1. Use more complex classifiers for the linear probe, such as a **small multi-layer perceptron ** instead of logistic regression.
2. Implement **hyperparameter search** (grid or random) for the linear probe classifier to ensure the model is tuned perfectly for the MNIST dataset.
**(all hyper parameter are used now are from openai/CLIP documentation)**

---