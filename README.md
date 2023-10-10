# EEGTransformer

The `EEGTransformer` class is a transformer-based architecture designed specifically for processing Electroencephalogram (EEG) data.

## Parameters

- `num_channels` (int): Specifies the number of channels in the EEG dataset.
- `num_timepoints` (int): Indicates the number of time points or the sequence length in the EEG data.
- `output_dim` (int): Defines the output dimensionality for the classifier layer.
- `hidden_dim` (int): Specifies the hidden layer dimensionality.
- `num_heads` (int): Determines the number of attention heads to be used in the multi-head self-attention mechanism.
- `key_query_dim` (int): Denotes the dimensionality for the key/query pairs in the self-attention mechanism.
- `hidden_ffn_dim` (int): Indicates the hidden layer dimensionality for the feed-forward network.
- `intermediate_dim` (int): Refers to the dimensionality of the intermediate layer in the feed-forward network.
- `ffn_output_dim` (int): Specifies the output size of the feed-forward network.

## Attributes

- `positional_encoding` (torch.Tensor): A tensor of shape `(num_channels, num_timepoints)` that imparts the sequence position information.
- `multihead_attn` (nn.MultiheadAttention): Implements the multi-head self-attention mechanism.
- `ffn` (nn.Sequential): Constructs a feed-forward network composed of a linear transformation followed by ReLU activation and another linear transformation.
- `norm1` and `norm2` (nn.LayerNorm): Execute layer normalization.
- `classifier` (nn.Linear): Deploys a final linear transformation layer to categorize the input into designated classes.

## Methods

- `forward(X)`: Outlines the forward propagation for the model.
    - `X` (torch.Tensor): The input tensor for EEG data, which should have a shape of `(batch_size, num_channels, num_timepoints)`.
    - **Steps**:
        1. Standardize the input tensor.
        2. Apply positional encoding.
        3. Implement multi-head self-attention.
        4. Reshape the attention output and apply layer normalization.
        5. Forward the data through the feed-forward network.
        6. Flatten the resultant tensor and direct it through a classifier layer.
        7. Yield the final output.

## Notes

- The model applies layer normalization after the multi-head self-attention and feed-forward network stages.
- Positional encoding is utilized to impart sequence position information to the model, which can either be relative or absolute.
- The classifier layer flattens the model output and categorizes it into `output_dim` classes.

## Usage

To employ the `EEGTransformer` model, instantiate the class using the desired parameters. Then, similar to any other PyTorch model, forward the input data to the model and utilize the returned output for either training or inference.

```python
# Sample Usage
model = EEGTransformer(num_channels=32, num_timepoints=200, output_dim=2,
                       hidden_dim=512, num_heads=8, key_query_dim=512,
                       hidden_ffn_dim=512, intermediate_dim=2048,
                       ffn_output_dim=32)

input_data = torch.randn(64, 32, 200)
output = model(input_data)

```

Ensure that the model is paired with a compatible loss function and optimizer for effective training. Depending on the specifics of the EEG dataset or application requirements, the model can be further refined.
