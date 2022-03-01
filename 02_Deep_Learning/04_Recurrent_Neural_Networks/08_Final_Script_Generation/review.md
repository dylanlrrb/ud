## Meets Specifications

Congratulations ![:tada:](/assets/images/emojis/tada.png ":tada:") ![:tada:](/assets/images/emojis/tada.png ":tada:") ![:tada:](/assets/images/emojis/tada.png ":tada:")

- Your submission reveals that you have made an **excellent effort** in finishing this project,especially the batching, model architecture and hyperparameters.
- Very good hyperparameters and decreasing cross entropy loss. It is great that you have got everything right in first review ![:thumbsup:](/assets/images/emojis/thumbsup.png ":thumbsup:")
- Please go through the additional suggestions in the rubric below.
- I wish you all the best for next adventures ![:rocket:](/assets/images/emojis/rocket.png ":rocket:")

**Few references to explore:**

- [Colah's Blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) A visual explanation of LSTMs you want to look at to understand it more.
- [Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) Andrej is director of AI and AutoPilot Vision at Tesla, this link explains the unreasonable effectiveness of RNN
- [Rohan Kapur](https://ayearofai.com/rohan-lenny-3-recurrent-neural-networks-10300100899b) , This link is an overall explanation of RNNs that you may find useful and insightful.

(feel free to reach out to mentors in the [knowledge forum](https://knowledge.udacity.com/) regarding any question or confusion)

Keep up the good work ![:thumbsup:](/assets/images/emojis/thumbsup.png ":thumbsup:") Stay Udacious ![:udacious:](/assets/images/emojis/udacious.png ":udacious:")

## All Required Files and Tests

The project submission contains the project notebook, called “dlnd\_tv\_script\_generation.ipynb”.

All required files are present ![:thumbsup:](/assets/images/emojis/thumbsup.png ":thumbsup:")

Bonus tips:

- It is recommended to export your conda environment into environment.yaml file using command **conda env export -f environment.yaml**, so that you can recreate your conda environment later.
- While submitting this to any version control system like Github, make sure to include helper, data and environment files and exclude and temp files. It will help you in future if you want to re-execute it. Some [guideline](https://udacity.github.io/git-styleguide/) for best practice.

All the unit tests in project have passed.

All the unit tests in project have passed. ![:thumbsup:](/assets/images/emojis/thumbsup.png ":thumbsup:")

Donald Knuth (a famous computer science pioneer) once famously said about unit tests:  
_“Beware of bugs in the above code; I have only proved it correct, not tried it.”_

Article on unit-test in machine learning system [here](https://www.jeremyjordan.me/testing-ml)

## Pre-processing Data

The function `create_lookup_tables` create two dictionaries:

- Dictionary to go from the words to an id, we'll call vocab\_to\_int
- Dictionary to go from the id to word, we'll call int\_to\_vocab

The function `create_lookup_tables` return these dictionaries as a tuple (vocab\_to\_int, int\_to\_vocab).

Good job! ![:clap:](/assets/images/emojis/clap.png ":clap:")

- The [Counter](https://pymotw.com/2/collections/counter.html) is a convenient way to get the information needed for that approach, but it has some extra overhead we don't need, so you could just use a `set` instead: set(text). The sorting is also unnecessary.
- You also only need to enumerate once, if you create both dicts **inside a single for loop**. All of these things will save some compute power/time.  
    Alternatively, it could be implemented like this:

```
vocab = set(text)
vocab_to_int, int_to_vocab = {}, {}
for i, w in enumerate(vocab):
    vocab_to_int[w] = i
    int_to_vocab[i] = w

return (vocab_to_int, int_to_vocab)
```

```
from collections import Counter

 word_counts = Counter(text)
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_to_int = dict()
    int_to_vocab = dict()

    for i, word in enumerate(sorted_words):
        vocab_to_int[word] = i
        int_to_vocab[i] = word
    return vocab_to_int, int_to_vocab
```

The function `token_lookup` returns a dict that can correctly tokenizes the provided symbols.

![:thumbsup:](/assets/images/emojis/thumbsup.png ":thumbsup:")

- Converting each punctuation into explicit token is very handy when working with RNNs.
- All 10 entries are present
- Do read up this [link](https://datascience.stackexchange.com/a/11421) to understand what other pre-processing steps are carried out before feeding text data to RNNs.

## Batching Data

The function `batch_data` breaks up word id's into the appropriate sequence lengths, such that only complete sequence lengths are constructed.

Good Job ![:clap:](/assets/images/emojis/clap.png ":clap:")

- The implementation breaking up word id's into the appropriate sequence lengths

In the function `batch_data`, data is converted into Tensors and formatted with TensorDataset.

It is recommended to add explanatory comments in between, look at this alternative implementation :

```
    # get number of targets we can make
    n_targets = len(words) - sequence_length
    # initialize feature and target 
    feature, target = [], []
    # loop through all targets we can make
    for i in range(n_targets):
        x = words[i : i+sequence_length]    # get some words from the given list
        y = words[i+sequence_length]        # get the next word to be the target
        feature.append(x)
        target.append(y)

    feature_tensor, target_tensor = torch.from_numpy(np.array(feature)), torch.from_numpy(np.array(target))
    # create data
    data = TensorDataset(feature_tensor, target_tensor)
    # create dataloader
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
     # return a dataloader
    return dataloader
```

This implementation is basically responsible for loading sequenced data into Tensors in order for PyTorch's TensorDataset utility to generate the dataset.

```
 data = TensorDataset(feature_tensors, target_tensors)
```

[Check this dataloading tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

Finally, `batch_data` returns a DataLoader for the batched training data.

The unit test output tensor verifies the implementation ![:thumbsup:](/assets/images/emojis/thumbsup.png ":thumbsup:")

- The function of Dataloader is to combine a dataset and a sampler, and finally provides an iterable over the given dataset. Feature like automatic batching are also supported. Check details of Dataloader [here](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- Adding`shuffle=True`in the dataloader is allowing you to add randomness in the training sequences.

## Build the RNN

The RNN class has complete `__init__`, `forward` , and `init_hidden` functions.

![:thumbsup:](/assets/images/emojis/thumbsup.png ":thumbsup:")

- `__init__`, `forward` and `init_hidden` functions are complete, a good model architecture
- it is recommended to remove unnecessary/unused methods from the code. e.g. `dropout` is not really required here.
    
- RNN implements an LSTM Layer, and initializes it appropriately.
    
    ```
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                              dropout=dropout, batch_first=True)
    ```
    
- we can also use GRU as well in place of LSTM:
    
    ```
    self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
    self.GRU = nn.GRU(self.embedding_dim,self.hidden_dim,self.n_layers,batch_first=True,dropout=self.dropout)
    self.fc = nn.Linear(self.hidden_dim,self.output_size)
    ```
    
- Check this article: [Difference between LSTM and GRU](https://datascience.stackexchange.com/questions/14581/when-to-use-gru-over-lstm)

The RNN must include an LSTM or GRU and at least one fully-connected layer. The LSTM/GRU should be correctly initialized, where relevant.

The ideal structure is as follows:

- Embedding layer (nn.Embedding) before the LSTM or GRU layer.
- The fully-connected layer comes at the end to get our desired number of outputs.
- It is also **recommended to not use a dropout after LSTM and before FC layer**, as the drop out is already incorporated in the LSTMs, A lot of students adds it and then end up finding convergence difficult.The added layer could cause the model to lose key information that's needed to improve the model's performance.

## RNN Training

- Enough epochs to get near a minimum in the training loss, no real upper limit on this. Just need to make sure the training loss is low and not improving much with more training.
- Batch size is large enough to train efficiently, but small enough to fit the data in memory. No real “best” value here, depends on GPU memory usually.
- Embedding dimension, significantly smaller than the size of the vocabulary, if you choose to use word embeddings
- Hidden dimension (number of units in the hidden layers of the RNN) is large enough to fit the data well. Again, no real “best” value.
- n\_layers (number of layers in a GRU/LSTM) is between 1-3.
- The sequence length (seq\_length) here should be about the size of the length of sentences you want to look at before you generate the next word.
- The learning rate shouldn’t be too large because the training algorithm won’t converge. But needs to be large enough that training doesn’t take forever.

![:rocket:](/assets/images/emojis/rocket.png ":rocket:")

- Enough epochs to get near a minimum in the training loss.
- Batch size is large enough to train efficiently. In order to use the GPU more efficiently, we can always try to set a value that is a power of two (e.g. 64 or 128 or 256)
- Sequence length is about the size of the length of sentences we want to generate. Considering the fact that there are approximately an average of 11.504 words per line and 15.248 sentences in each scene
- Size of embedding is in the range of \[200-300\]. The vocab contained ~46,367 unique words. Now you can try to cut this down significantly by 98% to 1000 embeddings. For example, Google's news word vectors, the GloVe vectors, and other word vectors are usually in the range 50 to 300
- Learning rate seems good based on other hyper parameter
- Hidden Dimension: 128-256 hidden dimensions to give the network a solid amount of features/states to learn from. [recommendation](https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046) on how to select Hidden Dimension `hidden_dim = 5*int(len(train_loader.dataset) / (embedding_dim + output_size))`
- Number of layers is in between 1-3 as suggested in the project.

> Your efforts shows that you have really have thought about it to get an optimized value ![:fire:](/assets/images/emojis/fire.png ":fire:")

The printed loss should decrease during training. The loss should reach a value lower than 3.5.

![:checkered_flag:](/assets/images/emojis/checkered_flag.png ":checkered_flag:") excellent decreasing loss ..

```
Epoch:    5/8     Loss: 3.5432490034103394

Epoch:    6/8     Loss: 3.4550640053895054

Epoch:    6/8     Loss: 3.401256357192993

Epoch:    6/8     Loss: 3.423488987445831

Epoch:    7/8     Loss: 3.318705863794502
```

There is a provided answer that justifies choices about model size, sequence length, and other parameters.

Detailed point-wise answer explaining the approach as well as reasoning behind the answer.

> The act of elaborating your approach often leads to a deeper understanding of the material ![:blush:](/assets/images/emojis/blush.png ":blush:")

## Generate TV Script

The generated script can vary in length, and should look structurally similar to the TV script in the dataset.

It doesn’t have to be grammatically correct or make sense.

well generated fun script! ![:clap:](/assets/images/emojis/clap.png ":clap:")

- all the lines are making sense
- sentences are grammatically intact