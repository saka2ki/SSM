# example
## StateSpaceModel * InputCopying
* python main.py model=StateSpaceModel data=InputCopying bsz=256 lr=0.001 epochs=10 device='cuda' model.init.vocab_size=null model.init.dim=512 model.init.N=64 model.init.layer=6 model.init.dropout=0. model.forward.cnn=True data.length=5 data.n_train=16384 data.n_test=1024
## StateSpaceModel * AssociateRecall
* python main.py model=StateSpaceModel data=AssociateRecall bsz=256 lr=0.001 epochs=10 device='cuda' model.init.vocab_size=null model.init.dim=512 model.init.N=64 model.init.layer=6 model.init.dropout=0.  model.forward.cnn=True data.length=5 data.n_train=16384 data.n_test=1024
## TransformerModel * InputCopying
* python main.py model=TransformerModel data=InputCopying bsz=256 lr=0.001 epochs=10 device='cuda' model.init.vocab_size=null model.init.dim=512 model.init.nhead=8 model.init.layer=6 model.init.dropout=0. data.length=5 data.n_train=16384 data.n_test=1024
## TransformerModel * AssociateRecall
* python main.py model=TransformerModel data=AssociateRecall bsz=256 lr=0.001 epochs=10 device='cuda' model.init.vocab_size=null model.init.dim=512 model.init.nhead=8  model.init.layer=6 model.init.dropout=0. data.length=5 data.n_train=16384 data.n_test=1024

# config
python main.py model=StateSpaceModel data=InputCopying bsz=256 lr=0.001 epochs=10 device='cuda' model.init.vocab_size=null model.init.dim=512 model.init.N=64 model.init.layer=6 model.init.dropout=0. model.forward.cnn=True data.length=64 data.n_train=16384 data.n_test=1024
 
python main.py model=StateSpaceModel data=AssociateRecall bsz=256 lr=0.001 epochs=10 device='cuda' model.init.vocab_size=null model.init.dim=512 model.init.N=64 model.init.layer=6 model.init.dropout=0. model.forward.cnn=True data.length=64 data.n_train=16384 data.n_test=1024

python main.py model=TransformerModel data=InputCopying bsz=256 lr=0.001 epochs=10 device='cuda' model.init.vocab_size=null model.init.dim=512 model.init.nhead=8 model.init.layer=6 model.init.dropout=0. data.length=64 data.n_train=16384 data.n_test=1024

python main.py model=TransformerModel data=AssociateRecall bsz=256 lr=0.001 epochs=10 device='cuda' model.init.vocab_size=null model.init.dim=512 model.init.nhead=8  model.init.layer=6 model.init.dropout=0. data.length=64 data.n_train=16384 data.n_test=1024
  
* python main.py model=StateSpaceModel model.init.vocab_size=null model.init.dim=16 model.init.N=64 model.init.layer=4 model.dropout=0. model.forward.cnn=True
* python main.py model=TransformerModel model.init.vocab_size=null model.init.dim=16 model.init.layer=4
* python main.py data=InputCopying data.length=5 data.n_train=8192 data.n_test=1024
* python main.py data=AssociateRecall data.length=5 data.n_train=8192 data.n_test=1024
* python main.py data=MNIST

## model
### StateSpaceModel
* vocab_size: vocabrary size of tokenizer for embedding layer
* dim: dimention size of input
* N: dimention of state-space
* layer: the number of layer of model
* dropout: rate of dropout probability
* cnn: True(cnn) or False(rnn)

### TransformerModel
* vocab_size: vocabrary size of tokenizer for embedding layer
* dim: dimention size of input
* layer: the number of layer of model
* dropout: rate of dropout probability

## dataset
### InputCopying
### AssociateRecall
### MNIST