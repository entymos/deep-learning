"""
Add below code in the "/youtube-8m-master/video_level_models.py"

You can run this code from console(terminal). I use Windows PowerShell.

python train.py --feature_names='mean_rgb,mean_audio' --feature_sizes='1024,128' --model='MyModel' --train_data_pattern=C:/Users/yoon/Desktop/youtube8m/yt8m/v2/video/train*.tfrecord --train_dir C:/Users/yoon/Desktop/youtube8m/yt8m/v2/models/video/sample_model --start_new_model

"""

def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a my model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    W1 = tf.Variable(tf.random_normal([1152,2048], stddev=0.01))
    L1 = tf.nn.relu(tf.matmul(model_input, W1))

    W2 = tf.Variable(tf.random_normal([2048,4096], stddev=0.01))
    L2 = tf.nn.relu(tf.matmul(L1, W2))

    W3 = tf.Variable(tf.random_normal([4096, vocab_size], stddev=0.01))
    output = tf.matmul(L2, W3)
    output = tf.nn.softmax(output)

    return {"predictions": output}
