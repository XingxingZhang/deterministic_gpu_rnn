
import theano
import theano.tensor as TT
import numpy, sys

def clip_grad(grads, norm, grad_clip):
    # clip the grads, when over a threshold
    _grads = []
    for g in grads:
        _grads.append( TT.switch(TT.ge(norm, grad_clip), g*grad_clip/norm, g) )
    
    return _grads

def SGD(params, grads, lr):
    return [(p, p - lr*g) for p, g in zip(params, grads)]

class RNNLM(object):
    
    def __init__(self, rng, settings):
        min_val = -settings['init_range']
        max_val = settings['init_range']
        n_vocab = settings['n_vocab']   # size of vocabulary
        n_in = settings['n_in']         # word embedding size
        n_hid = settings['n_hid']       # hidden layer size
        n_out = settings['n_vocab']     # this is a language model... 
        
        def rng_val(shape):
            return numpy.asarray(rng.uniform(min_val, max_val, size = shape), 
                                  dtype = theano.config.floatX)
        
        # word embedding matrix
        self.We = theano.shared(rng_val((n_vocab, n_in)), name = 'We')
        
        self.Wih = theano.shared(rng_val((n_in, n_hid)), name = 'Wih')
        self.Whh = theano.shared(rng_val((n_hid, n_hid)), name = 'Whh')
        self.Who = theano.shared(rng_val((n_hid, n_out)), name = 'Who')
        
        init_hid_val = settings['init_hid_val']
        # init hidden layer. set as params, so can be trained later
        init_val = numpy.asarray(numpy.ones((n_hid,))*init_hid_val, dtype = theano.config.floatX)
        self.h0 = theano.shared(value = init_val, name = 'h0')
        
        self.params = [
                       self.We,
                       self.Wih, self.Whh, self.Who,
                       self.h0
                       ]
        
        self.params_l2 = [
                          self.We,
                          self.Wih,
                          self.Whh,
                          self.Who
                          ]
        
        self.grad_clip = None if (not 'grad_clip' in settings or settings['grad_clip'] <= 0) else numpy.float32(settings['grad_clip'])
        self.l2_weight = None if (not 'L2' in settings or settings['L2'] <= 0) else numpy.float32(settings['L2'])
        self.algo = settings['algo']
        
        print '[SimpleRNNLM] alloc and init params done!'
    
    def fprop_batch(self, x):
        # x is 2D, shape (seqlen, bs)
        # word embedding
        emb = self.We[x]
        
        def one_step(x_t, h_tm1):
            h_t = TT.tanh( TT.dot(x_t, self.Wih) + TT.dot(h_tm1, self.Whh) )
            
            return h_t
        
        h0s = TT.alloc(self.h0, x.shape[1], self.h0.shape[0])
        
        init_outputs = h0s
        hs, _ = theano.scan(
                                  fn = one_step,
                                  sequences = [emb],
                                  outputs_info = init_outputs
                                  )
        y_a = TT.dot(hs, self.Who)
        y_given_x = TT.nnet.softmax( y_a.reshape( (y_a.shape[0]*y_a.shape[1], y_a.shape[2]) ) )
        
        return y_given_x.reshape( (y_a.shape[0], y_a.shape[1], y_given_x.shape[1]) )
    
    def build_batch(self):
        x = TT.imatrix('x')                # 2D int32
        x_mask = TT.fmatrix('x_mask')       # float32
        y = TT.imatrix('y')                 # 2D int32
        y_given_x = self.fprop_batch(x)     # 3D, shape (seq_len, bs, n_out)
        self.get_y_given_x = theano.function(inputs = [x], outputs = y_given_x)
        
        y_given_x_ = y_given_x.reshape((y_given_x.shape[0]*y_given_x.shape[1], y_given_x.shape[2]))
        y_ = y.reshape((y.shape[0]*y.shape[1], ))
        nll = -TT.sum( 
                       TT.log( y_given_x_[TT.arange(y_.shape[0]), y_] ) * 
                       x_mask.reshape( (x_mask.shape[0]*x_mask.shape[1], ) ) 
                       ) / x_mask.shape[1]  # nll is the sum of nll divided by batch size
        cost = nll
        
        # l2 norm cost
        if self.l2_weight is not None:
            L2 = 0
            for p in self.params_l2:
                L2 += TT.sum(p ** 2)
            cost += self.l2_weight * L2
            print '[SimpleRNNLM] L2 norm used %g' % self.l2_weight
        else:
            print '[SimpleRNNLM] L2 norm not used'
        
        lr = TT.scalar('lr')
        
        print '[SimpleRNNLM] ... get grads ...'
        grads = TT.grad(cost, self.params)
        grad_norm = TT.sqrt(sum([TT.sum(g**2) for g in grads]))
        if self.grad_clip is not None:
            grads = clip_grad(grads, grad_norm, self.grad_clip)
            grad_norm = TT.sqrt(sum([TT.sum(g**2) for g in grads]))
        else:
            print '[SimpleRNNLM] no grad_clip is used'
        print '[SimpleRNNLM] ... got grads ...'
        
        print '[SimpleRNNLM] algo = ', self.algo
        if self.algo == 'SGD':
            updates = SGD(self.params, grads, lr)
        else:
            sys.stderr.write('Not recognized training algorithm')
            sys.exit(1)
        
        print '[SimpleRNNLM] ...build training function...'
        self.train_batch_fn = theano.function(inputs = [x, x_mask, y, lr], outputs = nll, updates = updates)
        print '[SimpleRNNLM] ...build training function done...'
        
        # valid_fn return nll
        self.valid_batch_fn = theano.function(inputs = [x, x_mask, y], outputs = nll)
        
        # detailed valid function return both nll and y_given_x
        self.detailed_valid_batch_fn = theano.function(inputs = [x, x_mask, y], outputs = [nll, y_given_x])
        
        print '[SimpleRNNLM] build train_fn and valid_fn done!'
        
        return self.train_batch_fn, self.valid_batch_fn

