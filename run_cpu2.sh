
C_DIR=~/.theano_`hostname`

THEANO_FLAGS=device=cpu,floatX=float32,base_compiledir=$C_DIR python main.py 2>&1 | tee cpu2.txt

