### Machine Learning
##### To do:
- Supervised learning final
- unsupervised learning final
- siganl separation
- background removal (am I reshaping rather than viewing anywhere? need to speed it up)
- genomics PCA
##### Ideas:
- Brain wave separation
- expalinability layers in machine leaning models [1](https://towardsdatascience.com/explainable-ai-xai-lime-shap-two-great-candidates-to-help-you-explain-your-machine-learning-a95536a46c4e)

### Convolutional Neural Nets
Different tpyes of convolutions: https://medium.com/codex/7-different-convolutions-for-designing-cnns-that-will-level-up-your-computer-vision-project-fec588113a64

##### To do:
- CFIAR classification final
- tSNE on penultimate layer of landmark classifier
- elasticsearch with vector similarity of images (blog post)
- style transfer notebook, slow backprop method
- backprop painting [1](https://towardsdatascience.com/visual-interpretability-for-convolutional-neural-networks-2453856210ce), [2](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)
- Dermatologist AI
- cat/dog classifier in tensorflow (DEMO)
- Class activation Map/ Saliency Map/ Attribution Maps [1](https://mrsalehi.medium.com/a-review-of-different-interpretation-methods-in-deep-learning-part-1-saliency-map-cam-grad-cam-3a34476bc24d), [2](https://openaccess.thecvf.com/content/ICCV2021/papers/Ruiz_Generating_Attribution_Maps_With_Disentangled_Masked_Backpropagation_ICCV_2021_paper.pdf), [3](https://towardsdatascience.com/visual-interpretability-for-convolutional-neural-networks-2453856210ce)
- View at feature maps in a real time web app (DEMO) [1](https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573), [2](https://experiments.withgoogle.com/what-neural-nets-see), [3](https://discuss.pytorch.org/t/how-to-access-input-output-activations-of-a-layer-given-its-parameters-names/53772)
- Multi box detector using mobile net + demo app (first exporation in pytorch then implementationin a web app) [1](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection), [2](https://medium.com/axinc-ai/mobilenetssd-a-machine-learning-model-for-fast-object-detection-37352ce6da7d#:~:text=MobilenetSSD%20is%20an%20object%20detection,detection%20optimized%20for%20mobile%20devices), [3](https://adityakunar.medium.com/object-detection-with-ssd-and-mobilenet-aeedc5917ad0), [4](https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab), [5](https://medium.com/@techmayank2000/object-detection-using-ssd-mobilenetv2-using-tensorflow-api-can-detect-any-single-class-from-31a31bbd0691)
- Deep dream clone [1](https://www.alanzucconi.com/2015/07/06/live-your-deepdream-how-to-recreate-the-inceptionism-effect/), [2](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html), [3](https://www.youtube.com/watch?v=GHVaaHESrlY)

##### Ideas:
- style transfer pt 2, model per style to transfer [1](https://www.youtube.com/watch?v=y54wAlE04qU)
- style transfter pt 3, arbitrary style transfer in browser (DEMO) [1](https://www.youtube.com/watch?v=y54wAlE04qU), [2](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), [3](https://reiinakano.com/arbitrary-image-stylization-tfjs/), [4](https://magenta.tensorflow.org/blog/2018/12/20/style-transfer-js/), [5](https://arxiv.org/pdf/1705.06830.pdf)
- condenser networks
- create Skip connections like in resnet?

### Autoencoders
##### To Do:
- De-noising an image and or sound
- Add color to a BW image
- mnist to 3D volume [1](https://www.kaggle.com/daavoo/3d-mnist)
##### Ideas:
- Autoencoder for taking a 2d image of a cell and turning it into a 3d volume [1](https://stackoverflow.com/questions/47373421/from-2d-to-3d-using-convolutional-autoencoder), [2](https://www.allencell.org/3d-cell-viewer.html), [3](https://www.allencell.org/data-downloading.html)
- Image super resolution auto encoder
- Compare dimensionality reduction with autoencoders vs PCA
- What else are autoencoders good for?

### Recurrent Neural Nets
General Resources: [1](https://www.youtube.com/watch?v=iX5V1WpxxkY), [2](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), [3](http://blog.echen.me/2017/05/30/exploring-lstms/)

[Awesome-RNNs](https://github.com/kjw0612/awesome-rnn)

Sequence model type explaniations: [1](https://machinelearningmastery.com/models-sequence-prediction-recurrent-neural-networks/), [2](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html), [3](https://jinglescode.github.io/2020/05/21/three-types-sequence-prediction-problems/), [4](https://wandb.ai/ayush-thakur/dl-question-bank/reports/LSTM-RNN-in-Keras-Examples-of-One-to-Many-Many-to-One-Many-to-Many---VmlldzoyMDIzOTM)
##### To do:
- TV script (one to many)
- Word to vector (negative sampling explored) [1](https://gist.github.com/GavinXing/9954ea846072e115bb07d9758892382c), [2](https://stackoverflow.com/questions/62456558/is-one-hot-encoding-required-for-using-pytorchs-cross-entropy-loss-function)
- Sentiment analysis of movie reviews (sequence to one) [1](https://stackoverflow.com/questions/54892813/what-is-the-difference-between-sequence-to-sequence-and-sequence-to-one-regressi)
- project with seq2seq without attention (translation in order to compare to w/ attention?)
- Translation with Attention project (seq2seq w/ attention) + class activation map of attention matrix as sentence is translated [1](https://www.tensorflow.org/text/tutorials/nmt_with_attention), [2](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html), [3](https://www.quora.com/Why-in-a-seq2seq-RNN-model-do-we-ignore-the-output-of-the-encoder-and-just-pass-to-the-decoder-the-state-of-the-encoder), [4](http://blog.echen.me/2017/05/30/exploring-lstms/)
- Image captioning with attention, visualizing focused parts of image attention is given to [1](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)
##### Ideas:
- Projects based off of [Magenta demos](https://magenta.tensorflow.org/demos)
- sketch rnn made arbitrary in the same way as the arbitrary style transfer?
- Create images from text
- Video to pose estimation
- Transformer model with attention
- video description generator
- Summarization bot (look in hyperparameter reousrces for starting point)
- rnn transfer learning?
- rnn on biological data
- lip reading
- Screeps bot or other game
- predicting the next frame of video
- Hierachical MultiscaleRecurrent Neural Networks [1](https://openreview.net/pdf?id=S1di0sfgl), [2](https://medium.com/paper-club/hierarchical-multiscale-recurrent-neural-networks-9e614e4fb04)

### Generative Aveserial Neural Nets
##### To do:
##### Ideas:

### Reinforcement Learning 
##### To do:
##### Ideas:
- play atari games
- Trading bot [1](https://link.medium.com/4m9EqKdZSnb)
- Renforcment problems, multiagent? [1](https://neptune.ai/blog/best-benchmarks-for-reinforcement-learning), [2](https://www.reddit.com/r/MachineLearning/comments/cnrrh2/p_i_made_a_persistent_online_environment_for_ai/)


### Computer Vision
##### To do:
##### Ideas:

### Natural Language Processing
##### To do:
##### Ideas:

### Bioinformatics:
##### To Do:
##### Ideas:
- George holz covid 19 exploration
- See what you can gather from the vaccine repo

### Career Transition Resources:
- https://www.youtube.com/watch?v=7xkDGdnRLTQ&feature=youtu.be
- https://aqeel-anwar.medium.com/machine-learning-resume-that-got-me-shortlisted-for-meta-microsoft-nvidia-uber-samsung-intel-220761c1a850

### Interesting blogs or articles
- [Awesome-Deep-learning](https://github.com/ChristosChristofidis/awesome-deep-learning)
- https://medium.com/towards-data-science/deep-learning-with-magnetic-resonance-and-computed-tomography-images-e9f32273dcb5
- Multimodal neurons https://openai.com/blog/multimodal-neurons/
- https://medium.com/kaggle-blog
- https://towardsdatascience.com/dropout-on-convolutional-layers-is-weird-5c6ab14f19b2
- Evolutionary Feature Selection for Machine Learning https://towardsdatascience.com/evolutionary-feature-selection-for-machine-learning-7f61af2a8c12
- (linting of projects) https://stackoverflow.com/questions/58976685/type-check-jupyter-notebooks-with-mypy
https://medium.com/kaggle-blog/i-trained-a-model-what-is-next-d1ba1c560e26

















