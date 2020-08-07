# Mini-Attention

## A Keras Hirarchical Attention Layer for Document Classification in NLP :robot:

This library is an implementation of Heirarchical Attention Networks for Document Classification (Yang etal,2015).Link:[https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf0]. This is compatible with Keras and Tensorflow (keras version >=2.0.6). As the paper suggests, it uses hierarchical attention mechanism and capabilities of Word Encoder (including bi-directional Recurrent unit- GRU) ,Sentence Attention and Document Classification are addressed.

## Usability

The library or the Layer is compatible with Tensorflow and Keras. Installation is carried out using the pip command as follows:

```python
pip install MiniAttention==0.1
```

For using inside the Jupyter Notebook or Python IDE (along with Keras layers):

```python
import MiniAttention.MiniAttention as MA
```

The Layer takes as input a 3D Tensor with dimensions: (sample_size,steps,features)
The Layer as an output produces a 2D Tensor with dimensions: (sample_size,features)
This Layer can be used after the Keras.Embedding() Layer to provide a global attention using the features and Embedding weights.Additionally Embedding libraries like (Glove,Word2Vec,ELMO) can be used in the Embedding Layer. It can also be used in between and before LSTM/Bidirectional LSTM/ GRU and recurrent layers . In this context, it has a capability to cater to Sequential and Model (functional) types of Model architectures(specific for Keras). The functional (keras.models.Model) version is provided as follows:

```python

inp=Input(shape=(inp_shape,))
z=Embedding(max_features,256)(inp)
z=MA.MiniAttentionBlock(keras.initializers.he_uniform,None,None,keras.regularizers.L2(l2=0.02),None,None,None,None,None)(z)
z=tf.keras.layers.Bidirectional(LSTM(128,recurrent_activation="relu",return_sequences=True))(z)
z=tf.keras.layers.Bidirectional(LSTM(64,recurrent_activation="relu",return_sequences=True))(z)
z=MA.MiniAttentionBlock(keras.initializers.he_uniform,None,None,keras.regularizers.L2(l2=0.02),None,None,None,None,None)(z)
z=keras.layers.Dense(64,activation="relu")(z)
z=keras.layers.Dense(64,activation="relu")(z)
z=keras.layers.Dense(1,activation="sigmoid")(z)
model=keras.models.Model(inputs=inp,outputs=z)
model.compile(loss="binary_crossentropy",metrics=['accuracy'],optimizer=keras.optimizers.Adagrad(learning_rate=1e-3))
model.summary()
```

For Sequential Model (keras.models.Sequential):

```python
model = Sequential()
model.add(Embedding(max_features,128,input_shape=(100,)))
model.add(MA.MiniAttentionBlock(None,None,None,None,None,None,None,None,None))
model.add(LSTM(128))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='sigmoid'))
model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='Adagrad')
model.summary()
```

The arguments for the MiniAttentionBlock class include:

```
1.W_init: Weight Initializer - Compatible with keras.initializers
2.b_init: Bias Initializer - Compatible with keras.initializers
3.u_init: Output Initializer -Compatible with keras.initializers
4.W_reg: Weight Regularizer - Compatible with keras.regularizers
5.b_reg: Bias Regularizer - Compatible with keras.regularizers
6.u_reg: Output Regularizer -Compatible with keras.regularizers
7.W_const: Weight Constraint - Compatible with keras.constraints
8.b_const: Bias Constraint - Compatible with keras.constraints
9.u_const: Output Constraint -Compatible with keras.constraints
10.bias: Boolean - True/False- Whether to use bias (Optional)
```

There are 3 main functions inside the MiniAttentionBlock class. The "**init**" method is used for initializeing the weight , bias tensors for computation. The "attention_block" method is used for assigning the variables (tensors) and checking for the input tensor size. The "build_nomask" method is used for computing the attention modules.Uses tanh as the internal activation function with exponential normalization.Masking has not been added to the library yet.

## Example

For reference on how to use the library, a Jupyter Notebook sample is present in the repository: "MiniAttention_on_IMDB.ipynb".

This is a sample which uses this Layer with Keras.layers.Embedding() Layer in IMDB binary classification.It uses the default keras embedding which is followed from the official tutorial by Keras docs:https://keras.io/examples/nlp/bidirectional_lstm_imdb

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT
