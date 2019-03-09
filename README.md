## Wine Predictor

A simple feed-forward neural network classifier for wine varieties based on user-submitted descriptions.  Trained on the [Wine Reviews Dataset](https://www.kaggle.com/zynicide/wine-reviews) from Kaggle and implemented in Tensorflow.  Model details: multi-layer perceptron with one hidden layer of size 200.  Trained using softmax cross-entropy loss with Adam Optimizer on minibatches of size 100.

To avoid Python dependency headaches, start up a virtual environment and install all dependencies using `pip install -r requirements.txt`.

Run **prepare_data.py** to load the data, split it into train, dev, and test sets, and get Numpy array representations for all input.  The input vectors will be saved to the disk to save time in future model experimentation.

To train the model and predict wine varieties for the test data, run **wine_predictor.py**

**To do:** write precision, recall, F1 evaluation script

### Cheers!

<pre>
                __
               )==(
               )==(
               |H |
               |H |
               |H |
              /====\
             /      \
            /========\
           :HHHHHHHH H:
           |HHHHHHHH H|
           |HHHHHHHH H|
           |HHHHHHHH H|
    |______|=|========|________|
     \     :/oO/      |\      /
      \    / oOOO aged| \    /
       \__/| OOO  wine|  \__/
        )( |  O       |   )(
        )( |==========|   )(
        )( |HHHHHHHH H|   )(
        )( |HHHHHHHH H|   )(
       .)(.|HHHHHHHH H|  .)(.
      ~~~~~~~~~~~~~~~~  ~~~~~~
    ------------------------------------------------
</pre>
<https://asciiart.website/index.php?art=objects/bottles>
