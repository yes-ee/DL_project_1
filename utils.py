import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x > 0] = 1
    return grad

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def cross_entropy_error(y, t):
    return -np.sum(t * np.log(y + 1e-7))

def binary_cross_entropy_error(y, t):
    epsilon = 1e-7
    return -np.mean(t * np.log(y + epsilon) + (1 - t) * np.log(1 - y + epsilon))

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters:
    input_data : 4D array (batch size, channels, height, width)
    filter_h : Height of the filter
    filter_w : Width of the filter
    stride : Stride size
    pad : Padding size

    Returns:
    col : 2D array (N * output height * output width, C * filter height * filter width)
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # Padding the input image
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # Fill in the columns for each filter region
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters:
    col : 2D array (flattened)
    input_shape : Shape of the original input data (N, C, H, W)
    filter_h : Height of the filter
    filter_w : Width of the filter
    stride : Stride size
    pad : Padding size

    Returns:
    img : Reconstructed 4D array from the column matrix
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def evaluate(model, images, labels):
    correct_count = 0
    total_count = images.shape[0]
    batch_size = 64
    epoch_loss = 0

    for i in range(0, total_count, batch_size):
        x_batch = images[i:i + batch_size]
        t_batch = labels[i:i + batch_size].reshape(-1, 1)

        # Forward pass
        out = model.predict(x_batch)
        preds_binary = (out > 0.5).astype(int)
        correct_count += np.sum(preds_binary == t_batch)

        # Calculate loss
        loss = binary_cross_entropy_error(out, t_batch)
        epoch_loss += loss

    accuracy = correct_count / total_count
    avg_loss = epoch_loss / (total_count // batch_size)
    return accuracy, avg_loss