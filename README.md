# The Impact of Feature Representation on the Accuracy of Photonic Neural Networks

This repository contains the data supporting the findings of my paper, as well as the code to generate it.

## Structure

The files are divided into two main sections:

- ```n-sphere/``` and ```/post_processing_nsphere.ipynb``` : These files correspond to the training of Artificial Neural Networks (ANNs) on an n-sphere dataset, examining the impact of different encoding functions.

- ```iris/``` and ```/post_processing_iris.ipynb``` These files correspond to the training of Photonic Neural Networks (PNNs) on the Iris dataset, examining the impact of different encoding functions.

## How to run

### Generating Plots
To generate the plots featured in the paper, simply run the provided notebooks. These will utilize pre-generated data.

### Create New Data

- **Iris Dataset:**
  - Navigate to the `iris/` folder.
  - Ensure all requirements are installed.
  - Run the `train_pnns.py` file using the `photontorch` package to create and train several PNNs on the Iris dataset:
    ```sh
    cd iris
    python train_pnns.py
    ```

- **n-sphere Dataset:**
  - Open and run the `experiments.ipynb` notebook to create the dataset and train various ANNs to classify its points.
  - Copy the results from this notebook to `post_processing_nsphere.ipynb` to generate the corresponding plots.

## Requirements

Please make sure to install all the necessary dependencies specified in the `requirements.txt` file before running the scripts and notebooks.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE 3 - see the [LICENSE](LICENSE) file for details.
