# EEGTransformer: Transformer-Based Model for EEG Classification

This repository presents `EEGTransformer`, a Transformer-based architecture tailored for processing Electroencephalogram (EEG) data. The model leverages self-attention mechanisms to capture intricate spatial and temporal dependencies inherent in EEG signals, facilitating effective classification tasks.

## Features

* **Transformer Architecture**: Utilizes multi-head self-attention to model complex relationships in EEG data.
* **Customizable Parameters**: Offers flexibility through adjustable hyperparameters to suit various EEG datasets.
* **End-to-End Pipeline**: Includes data preprocessing, model training, and evaluation within a cohesive framework.
* **Visualization Tools**: Provides tools for visualizing attention weights and model performance metrics.([github.com][2])

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/berdakh/Transformers-for-EEG-classification.git
   cd Transformers-for-EEG-classification
   ```



2. **Create a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```



3. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```



*Note: Ensure that [PyTorch](https://pytorch.org/) is installed, as it's central to the functionalities provided.*

## Usage

The primary script for training and evaluating the model is `EEGTransformer_Class.ipynb`. This Jupyter Notebook demonstrates the complete workflow, including data loading, model instantiation, training, and evaluation.([github.com][1], [github.com][3])

*To execute the notebook:*

1. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```



2. Open `EEGTransformer_Class.ipynb` and follow the cells sequentially.([github.com][1])

## Model Architecture

The `EEGTransformer` class is designed with the following components:([github.com][1])

* **Input Parameters**:

  * `num_channels` (int): Number of EEG channels.
  * `num_timepoints` (int): Number of time points in the EEG data.
  * `output_dim` (int): Dimensionality of the output layer.
  * `hidden_dim` (int): Dimensionality of the hidden layers.
  * `num_heads` (int): Number of attention heads in the multi-head attention mechanism.
  * `key_query_dim` (int): Dimensionality of the key and query vectors.
  * `hidden_ffn_dim` (int): Dimensionality of the hidden layer in the feed-forward network.
  * `intermediate_dim` (int): Dimensionality of the intermediate layer.
  * `ffn_output_dim` (int): Output dimensionality of the feed-forward network.([github.com][1])

* **Layers**:

  * `positional_encoding` (torch.Tensor): Adds positional information to the input data.
  * `multihead_attn` (nn.MultiheadAttention): Applies multi-head self-attention mechanism.
  * `ffn` (nn.Sequential): Feed-forward network comprising linear transformations and ReLU activations.
  * `norm1` and `norm2` (nn.LayerNorm): Layer normalization for stabilizing the learning process.([github.com][1])

* **Forward Method**:

  * Standardizes the input tensor.
  * Applies positional encoding.
  * Implements multi-head self-attention.
  * Applies layer normalization and feed-forward network.
  * Outputs the final classification predictions.([github.com][1], [scispace.com][4])

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments

This repository was developed by [Berdakh Abibullaev](https://github.com/berdakh), focusing on the application of Transformer models for EEG signal classification.

---

*For detailed explanations and methodologies, refer to the `EEGTransformer_Class.ipynb` notebook included in the repository.*
 
