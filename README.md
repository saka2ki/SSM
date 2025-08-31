# example
* python main.py model=StateSpaceModel data=InputCopying bsz=256 lr=0.001 epochs=10 device='cuda' model.init.vocab_size=null model.init.dim=512 model.init.div=512 model.init.N=64 model.init.layer=6 model.init.dropout=0. model.forward.cnn=True model.forward.is_emb=True model.forward.is_ssm=True model.forward.is_ffn=True data.length=5 data.n_train=16384 data.n_test=1024
* python main.py model=TransformerModel data=InputCopying bsz=256 lr=0.001 epochs=10 device='cuda' model.init.vocab_size=null model.init.dim=512 model.init.layer=6 data.length=5 data.n_train=16384 data.n_test=1024

# config
* python main.py model=StateSpaceModel data=InputCopying bsz=128 lr=0.001 epochs=1 device='cuda'
  
* python main.py model=StateSpaceModel model.init.vocab_size=null model.init.dim=16 model.init.div=16 model.init.N=64 model.init.layer=4 model.dropout=0. model.forward.cnn=True model.forward.is_emb=True model.forward.is_ssm=True model.forward.is_ffn=True
* python main.py model=TransformerModel model.init.vocab_size=null model.init.dim=16 model.init.layer=4
* python main.py data=InputCopying data.length=5 data.n_train=8192 data.n_test=1024
* python main.py data=AssociateRecall data.length=5 data.n_train=8192 data.n_test=1024
* python main.py data=MNIST

## model
### StateSpaceModel
* vocab_size: vocabrary size of tokenizer for embedding layer
* dim: dimention size of input
* div: the number of state-space, dim=dim//div
* N: dimention of state-space
* layer: the number of layer of model
* dropout: rate of dropout probability
* cnn: True(cnn) or False(rnn)
* is_emb, is_ssm, is_ffn: if you except these layers, you should set False

### TransformerModel
* vocab_size: vocabrary size of tokenizer for embedding layer
* dim: dimention size of input
* layer: the number of layer of model
* dropout: rate of dropout probability

## dataset
### InputCopying
### AssociateRecall
### MNIST