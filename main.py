
import numpy, cPickle, gzip, pprint
from RNN import RNNLM

pp = pprint.PrettyPrinter(indent=4)

def get_hyper_params():
    state = {}
    state['seed'] = 123
    state['init_range'] = 0.2
    state['n_vocab'] = 0
    state['n_in'] = 100
    state['n_hid'] = 100
    state['init_hid_val'] = 0.01
    state['grad_clip'] = 1.0
    state['algo'] = 'SGD'
    
    state['lr'] = 0.1
    state['bs'] = 20
    state['max_epoch'] = 20
    
    state['dataset_path'] = 'data/data'
    state['vocab_path'] = 'data/vocab.pkl.gz'
    
    return state

def to_std_batch(xs, eos):
    bs = len(xs)
    maxn = max([x.shape[0] for x in xs]) + 1
    x = numpy.zeros((maxn, bs), dtype = 'int32')
    x_mask = numpy.zeros((maxn, bs), dtype = 'float32')
    y = numpy.zeros((maxn, bs), dtype = 'int32')
    x[0, :] = eos
    for idx in range(bs):
        sx = xs[idx]
        x[1:sx.shape[0] + 1, idx] = sx
        x_mask[0:sx.shape[0] + 1, idx] = 1.
        y[0:sx.shape[0], idx] = sx
        y[sx.shape[0], idx] = eos
    return x, x_mask, y

def create_batch(data_file, label, vocab, bs):
    word2idx = vocab['word2idx']
    eos = word2idx['###root###']
    cnt = 0
    batch = []
    for line in open(data_file + '.' + label):
        words = line.strip().split()
        wids = numpy.asarray([word2idx.get(word, vocab['UNK']) for word in words], dtype = 'int32')
        batch.append(wids)
        cnt = cnt + 1
        if cnt % bs == 0:
            yield to_std_batch(batch, eos)
            del batch[:]
    
    if len(batch) != 0:
        yield to_std_batch(batch, eos)

def main():
    state = get_hyper_params()
    rng = numpy.random.RandomState(state['seed'])
    vocab = cPickle.load(gzip.open(state['vocab_path']))
    state['n_vocab'] = len(vocab['word2idx'])
    print 'load vocab done'
    print vocab.keys()
    print vocab['UNK']
    
    pp.pprint(state)
    
    rnn = RNNLM(rng, state)
    rnn.build_batch()
    
    epoch = 1
    lr = state['lr']
    while epoch <= state['max_epoch']:
        print 'epoch %d'%epoch
        # train
        train_iter = create_batch(state['dataset_path'], 'train', vocab, state['bs'])
        for x, x_mask, y in train_iter:
            rnn.train_batch_fn(x, x_mask, y, lr)
        
        valid_iter = create_batch(state['dataset_path'], 'valid', vocab, state['bs'])
        valid_cost = 0.0
        valid_n, cnt = 0, 0
        for x, x_mask, y in valid_iter:
            nll = rnn.valid_batch_fn(x, x_mask, y)
            valid_cost += x_mask.shape[1] * nll
            valid_n += x_mask.sum()
            cnt += x_mask.shape[1]
        ppl = 2 ** (valid_cost / numpy.log(2.0) / valid_n)
        entropy = valid_cost/valid_n
        print 'valid PPL %f ENTROPY %f' % (ppl, entropy)
        epoch = epoch + 1

if __name__ == '__main__':
    main()
