import random
import numpy as np
import sys
sys.setrecursionlimit(6000)
import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode

# import lasagne
# from lasagne import layers
# from lasagne import nonlinearities
import cPickle as pickle

import utils
import nn_utils

import copy

floatX = theano.config.floatX

class DMN_batch:
    
    def __init__(self, babi_train_raw, babi_test_raw, word2vec, word_vector_size, dim, 
                mode, answer_module, input_mask_mode, memory_hops, batch_size, l2,
                normalize_attention, batch_norm, dropout, **kwargs):
        
        print "==> not used params in DMN class:", kwargs.keys()
        
        self.vocab = {}
        self.ivocab = {}
        #build vocab
        self.vocab_build(babi_train_raw, babi_test_raw) # true word index starts from 1, 0 means zero pad
        self.word2vec = word2vec
        self.word_vector_size = word_vector_size
        self.dim = dim
        self.mode = mode
        self.answer_module = answer_module
        self.input_mask_mode = input_mask_mode
        self.memory_hops = memory_hops
        self.batch_size = batch_size
        self.l2 = l2
        self.normalize_attention = normalize_attention
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.train_input, self.train_q, self.train_answer, self.train_fact_count, self.train_input_mask = self._train_rawinput_toid(babi_train_raw)
        self.test_input, self.test_q, self.test_answer, self.test_fact_count, self.test_input_mask = self._test_rawinput_toid(babi_test_raw)
        self.vocab_size = len(self.vocab)+1

        rand_values=np.random.normal(0.0, 0.1, (self.vocab_size, self.word_vector_size))
        rand_values[0]=np.array(np.zeros(self.word_vector_size),dtype=theano.config.floatX)
        rand_values=self.load_word2vec_to_init(rand_values)        
        self.emb=theano.shared(name='emb', value=rand_values.astype(theano.config.floatX), borrow=True)
        #rand_values[0]=numpy.array([1e-50]*emb_size)


        self.inp_ids =  T.imatrix('inp_ids')
        self.q_ids = T.imatrix('q_ids')
#         self.input_var = T.tensor3('input_var') # (batch_size, seq_len, glove_dim)
#         self.q_var = T.tensor3('question_var') # as self.input_var
        self.answer_var = T.imatrix('answer_var') # (batch_size, anslen_max)
        self.fact_count_var = T.ivector('fact_count_var') # number of facts in the example of minibatch
        self.input_mask_var = T.imatrix('input_mask_var') # (batch_size, indices) 
        
        self.input_var=self.emb[self.inp_ids.flatten()].reshape((self.batch_size,self.inp_ids.shape[1], self.word_vector_size))
        self.q_var=self.emb[self.q_ids.flatten()].reshape((self.batch_size,self.q_ids.shape[1], self.word_vector_size))
        
        print "==> building input module"
        self.W_inp_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inp_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inp_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        input_var_shuffled = self.input_var.dimshuffle(1, 2, 0)
        inp_dummy = theano.shared(np.zeros((self.dim, self.batch_size), dtype=floatX))
        inp_c_history, _ = theano.scan(fn=self.input_gru_step, 
                            sequences=input_var_shuffled,
                            outputs_info=T.zeros_like(inp_dummy))
        
        inp_c_history_shuffled = inp_c_history.dimshuffle(2, 0, 1)
        
        inp_c_list = []
        inp_c_mask_list = []
        for batch_index in range(self.batch_size):
            taken = inp_c_history_shuffled[batch_index].take(self.input_mask_var[batch_index, :self.fact_count_var[batch_index]], axis=0)
            inp_c_list.append(T.concatenate([taken, T.zeros((self.input_mask_var.shape[1] - taken.shape[0], self.dim), floatX)]))
            inp_c_mask_list.append(T.concatenate([T.ones((taken.shape[0],), np.int32), T.zeros((self.input_mask_var.shape[1] - taken.shape[0],), np.int32)]))
        
        self.inp_c = T.stack(inp_c_list).dimshuffle(1, 2, 0)
        inp_c_mask = T.stack(inp_c_mask_list).dimshuffle(1, 0)
        
        q_var_shuffled = self.q_var.dimshuffle(1, 2, 0)
        q_dummy = theano.shared(np.zeros((self.dim, self.batch_size), dtype=floatX))
        q_q_history, _ = theano.scan(fn=self.input_gru_step, 
                            sequences=q_var_shuffled,
                            outputs_info=T.zeros_like(q_dummy))
        self.q_q = q_q_history[-1]
        
        
        print "==> creating parameters for memory module"
        self.W_mem_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_mem_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_mem_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_b = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_1 = nn_utils.normal_param(std=0.1, shape=(self.dim, 7 * self.dim + 0))
        self.W_2 = nn_utils.normal_param(std=0.1, shape=(1, self.dim))
        self.b_1 = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        self.b_2 = nn_utils.constant_param(value=0.0, shape=(1,))
        

        print "==> building episodic memory module (fixed number of steps: %d)" % self.memory_hops
        memory = [self.q_q.copy()]
        for iter in range(1, self.memory_hops + 1):
            current_episode = self.new_episode(memory[iter - 1])
            memory.append(self.GRU_update(memory[iter - 1], current_episode,
                                          self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                                          self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                                          self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid))                         
        '''
        last_mem_raw = memory[-1].dimshuffle((1, 0)) #(batch_size, dim)
         
        net = layers.InputLayer(shape=(self.batch_size, self.dim), input_var=last_mem_raw)
        if self.batch_norm:
            net = layers.BatchNormLayer(incoming=net)
        if self.dropout > 0 and self.mode == 'train':
            net = layers.DropoutLayer(net, p=self.dropout)
        last_mem = layers.get_output(net).dimshuffle((1, 0)) #(dim, batch_size)
        '''
        last_mem=memory[-1] #by wenpeng

        
        print "==> building answer module"
        self.W_a = nn_utils.normal_param(std=0.1, shape=(self.vocab_size, self.dim))
        
        if self.answer_module == 'feedforward':
            self.prediction = nn_utils.softmax(T.dot(self.W_a, last_mem)) #softmax over each column
        
        elif self.answer_module == 'recurrent':
            self.W_ans_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
            
            self.W_ans_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
            
            self.W_ans_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
            
            def answer_step(prev_a, prev_y):
                a = self.GRU_update(prev_a, T.concatenate([prev_y, self.q_q]),
                                  self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                                  self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                                  self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid)
                
                y = nn_utils.softmax(T.dot(self.W_a, a))
                return [a, y]
            
            # TODO: add conditional ending
            dummy = theano.shared(np.zeros((self.vocab_size, self.batch_size), dtype=floatX))
            results, updates = theano.scan(fn=self.answer_step,
                outputs_info=[last_mem, T.zeros_like(dummy)], #(last_mem, y)
                n_steps=1)
            self.prediction = results[1][-1]
        
        else:
            raise Exception("invalid answer_module")
        
        self.prediction = self.prediction.dimshuffle(1, 0)
        
        #wenpeng, add word embeddings as para        
        self.params = [self.W_inp_res_in, self.W_inp_res_hid, self.b_inp_res, 
                  self.W_inp_upd_in, self.W_inp_upd_hid, self.b_inp_upd,
                  self.W_inp_hid_in, self.W_inp_hid_hid, self.b_inp_hid,
                  self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                  self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                  self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid, #self.W_b
                  self.W_1, self.W_2, self.b_1, self.b_2, self.W_a,
                  self.emb]
        
        if self.answer_module == 'recurrent':
            self.params = self.params + [self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                              self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                              self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid]
                              
                              
        print "==> building loss layer and computing updates"
#         self.loss_ce = T.nnet.categorical_crossentropy(self.prediction, self.answer_var).mean()
        ii = T.repeat(T.arange(self.prediction.shape[0]), self.answer_var.shape[1])
        jj = self.answer_var.flatten()
        self.loss_ce = - T.sum(T.log(self.prediction[ii,jj]))
            
        if self.l2 > 0:
            self.loss_l2 = self.l2 * nn_utils.l2_reg(self.params)
        else:
            self.loss_l2 = 0
        
        self.loss = self.loss_ce + self.loss_l2
            
#         updates = lasagne.updates.adadelta(self.loss, self.params)
        #updates = lasagne.updates.momentum(self.loss, self.params, learning_rate=0.001)
        
        if self.mode == 'train':
            gradient = T.grad(self.loss, self.params)
            accumulator=[]
            for para_i in self.params:
                eps_p=np.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
                accumulator.append(theano.shared(eps_p, borrow=True))
#             updates = []
#             for param_i, grad_i in zip(self.params, gradient):
#                 updates.append((param_i, param_i - 0.005 * grad_i))   #AdaGrad  
            updates = []   
         
            for param_i, grad_i, acc_i in zip(self.params, gradient, accumulator):
                acc = acc_i + T.sqr(grad_i)
                if param_i == self.emb:
                    updates.append((param_i, T.set_subtensor((param_i - 0.5 * grad_i / T.sqrt(acc+1e-8))[0], theano.shared(np.zeros(self.word_vector_size)))))   #AdaGrad
                else:
                    updates.append((param_i, param_i - 0.5 * grad_i / T.sqrt(acc+1e-8)))   #AdaGrad
                updates.append((acc_i, acc))           
            print "==> compiling train_fn"
            self.train_fn = theano.function(inputs=[self.inp_ids, self.q_ids, self.answer_var, 
                                                    self.fact_count_var, self.input_mask_var], 
                                            outputs=[self.prediction, self.loss],
                                            updates=updates)
        
        print "==> compiling test_fn"
        self.test_fn = theano.function(inputs=[self.inp_ids, self.q_ids, #self.answer_var, do not use answer as input for dev and test
                                               self.fact_count_var, self.input_mask_var],
                                       outputs=self.prediction, on_unused_input='ignore')
        
    
    
    def GRU_update(self, h, x, W_res_in, W_res_hid, b_res,
                         W_upd_in, W_upd_hid, b_upd,
                         W_hid_in, W_hid_hid, b_hid):
        """ mapping of our variables to symbols in DMN paper: 
        W_res_in = W^r
        W_res_hid = U^r
        b_res = b^r
        W_upd_in = W^z
        W_upd_hid = U^z
        b_upd = b^z
        W_hid_in = W
        W_hid_hid = U
        b_hid = b^h
        """
        z = T.nnet.sigmoid(T.dot(W_upd_in, x) + T.dot(W_upd_hid, h) + b_upd.dimshuffle(0, 'x'))
        r = T.nnet.sigmoid(T.dot(W_res_in, x) + T.dot(W_res_hid, h) + b_res.dimshuffle(0, 'x'))
        _h = T.tanh(T.dot(W_hid_in, x) + r * T.dot(W_hid_hid, h) + b_hid.dimshuffle(0, 'x'))
        return z * h + (1 - z) * _h
    
    
    def _empty_word_vector(self): # used to pad shorter context or question
        return np.zeros((self.word_vector_size,), dtype=floatX)
    
    
    def input_gru_step(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_inp_res_in, self.W_inp_res_hid, self.b_inp_res, 
                                     self.W_inp_upd_in, self.W_inp_upd_hid, self.b_inp_upd,
                                     self.W_inp_hid_in, self.W_inp_hid_hid, self.b_inp_hid)
    
    
    def new_attention_step(self, ct, prev_g, mem, q_q):  # different with dmn_basic
        z = T.concatenate([ct, mem, q_q, ct * q_q, ct * mem, (ct - q_q) ** 2, (ct - mem) ** 2], axis=0)
        
        l_1 = T.dot(self.W_1, z) + self.b_1.dimshuffle(0, 'x')
        l_1 = T.tanh(l_1)
        l_2 = T.dot(self.W_2, l_1) + self.b_2.dimshuffle(0, 'x')
        G = T.nnet.sigmoid(l_2)[0]
        return G
        
        
    def new_episode_step(self, ct, g, prev_h):
        gru = self.GRU_update(prev_h, ct,
                             self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                             self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                             self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid)
        
        h = g * gru + (1 - g) * prev_h
        return h
     
       
    def new_episode(self, mem):
        g, g_updates = theano.scan(fn=self.new_attention_step,
            sequences=self.inp_c,
            non_sequences=[mem, self.q_q],
            outputs_info=T.zeros_like(self.inp_c[0][0])) 
        
        if (self.normalize_attention):
            g = nn_utils.softmax(g)
        
        e, e_updates = theano.scan(fn=self.new_episode_step,
            sequences=[self.inp_c, g],
            outputs_info=T.zeros_like(self.inp_c[0]))
        
        e_list = []
        for index in range(self.batch_size):
            e_list.append(e[self.fact_count_var[index] - 1, :, index])
        return T.stack(e_list).dimshuffle((1, 0))
   
   
    def save_params(self, file_name, epoch, **kwargs):
        with open(file_name, 'w') as save_file:
            pickle.dump(
                obj = {
                    'params' : [x.get_value() for x in self.params],
                    'epoch' : epoch, 
                    'gradient_value': (kwargs['gradient_value'] if 'gradient_value' in kwargs else 0)
                },
                file = save_file,
                protocol = -1
            )
    
    
    def load_state(self, file_name):
        print "==> loading state %s" % file_name
        with open(file_name, 'r') as load_file:
            dict = pickle.load(load_file)
            loaded_params = dict['params']
            for (x, y) in zip(self.params, loaded_params):
                x.set_value(y)
    
    
    def _process_batch(self, _inputs, _questions, _answers, _fact_counts, _input_masks, _mode):
        
        if _mode =="train":
            inputs = copy.deepcopy(_inputs)
            questions = copy.deepcopy(_questions)
            answers = copy.deepcopy(_answers)
            fact_counts = copy.deepcopy(_fact_counts)
            input_masks = copy.deepcopy(_input_masks)
            
            zipped = zip(inputs, questions, answers, fact_counts, input_masks)
            
            max_inp_len = 0
            max_q_len = 0
            max_fact_count = 0
            max_ans_len=0
            for inp, q, ans, fact_count, input_mask in zipped:
                max_inp_len = max(max_inp_len, len(inp))
                max_q_len = max(max_q_len, len(q))
                max_fact_count = max(max_fact_count, fact_count)
                max_ans_len = max(max_ans_len, len(ans))
            
            questions = []
            inputs = []
            answers = []
            fact_counts = []
            input_masks = []
            
            for inp, q, ans, fact_count, input_mask in zipped:
                while(len(inp) < max_inp_len):
#                     inp.append(self._empty_word_vector())
                    inp.append(0)
                
                while(len(q) < max_q_len):
#                     q.append(self._empty_word_vector())
                    q.append(0)
        
                while(len(input_mask) < max_fact_count):
                    input_mask.append(-1)
                
                
                rep_ans=max_ans_len/len(ans)
#                 print 'ans:', ans, 'rep_ans:', rep_ans
                ans=(ans*(rep_ans+1))[:max_ans_len]
    #             while(len(ans) < max_ans_len):
    #                 ans.append(-1)  # append word index -1 at the end of ans sequence
                
                #only change the inp, q, input_mask
                inputs.append(inp)
                questions.append(q)
                answers.append(ans)
                fact_counts.append(fact_count)
                input_masks.append(input_mask)
                
            inputs = np.array(inputs).astype(np.int32)
            questions = np.array(questions).astype(np.int32)
            answers = np.array(answers).astype(np.int32)
            fact_counts = np.array(fact_counts).astype(np.int32)
            input_masks = np.array(input_masks).astype(np.int32)
    
            return inputs, questions, answers, fact_counts, input_masks 
        if _mode =="test":
            inputs = copy.deepcopy(_inputs)
            questions = copy.deepcopy(_questions)
            fact_counts = copy.deepcopy(_fact_counts)
            input_masks = copy.deepcopy(_input_masks)
            
            zipped = zip(inputs, questions, fact_counts, input_masks)
            
            max_inp_len = 0
            max_q_len = 0
            max_fact_count = 0
            for inp, q, fact_count, input_mask in zipped:
                max_inp_len = max(max_inp_len, len(inp))
                max_q_len = max(max_q_len, len(q))
                max_fact_count = max(max_fact_count, fact_count)
            
            questions = []
            inputs = []
            fact_counts = []
            input_masks = []
            
            for inp, q, fact_count, input_mask in zipped:
                while(len(inp) < max_inp_len):
                    inp.append(0)
                
                while(len(q) < max_q_len):
                    q.append(0)
        
                while(len(input_mask) < max_fact_count):
                    input_mask.append(-1)

                
                #only change the inp, q, input_mask
                inputs.append(inp)
                questions.append(q)
                fact_counts.append(fact_count)
                input_masks.append(input_mask)
                
            inputs = np.array(inputs).astype(np.int32)
            questions = np.array(questions).astype(np.int32)
            fact_counts = np.array(fact_counts).astype(np.int32)
            input_masks = np.array(input_masks).astype(np.int32)
    
            return inputs, questions, fact_counts, input_masks
    
    def vocab_build(self, train_raw, test_raw):
        data_list=[train_raw, test_raw]
        for i in [0,1]:
            data_raw=data_list[i]
            for x in data_raw:
                inp = x["C"].split()   
    #             inp = [w for w in inp if len(w) > 0]
                q = x["Q"].split()
                if i==0:#train
                    ans= x["A"].strip().split()
 
                else:
                    ansSet= x["A"]
                    ans=[]
                    for single_ans in ansSet:
                        ans_wordlist=single_ans.strip().split()
                        ans+=ans_wordlist
    #             q = [w for w in q if len(w) > 0]
                
                
                for w in inp:
                    if not w in self.vocab: 
                        next_index = len(self.vocab)+1
                        self.vocab[w] = next_index
                        self.ivocab[next_index] = w   
                for w in q:
                    if not w in self.vocab: 
                        next_index = len(self.vocab)+1
                        self.vocab[w] = next_index
                        self.ivocab[next_index] = w    
                for w in ans:
                    if not w in self.vocab: 
                        next_index = len(self.vocab)+1
                        self.vocab[w] = next_index
                        self.ivocab[next_index] = w      
        print '==> vocab build over, totally:', len(self.vocab),' words'           
       
    def _train_rawinput_toid(self, data_raw):
        questions = []
        inputs = []
        answers = []
        fact_counts = []
        input_masks = []
        
        for x in data_raw:
            inp = x["C"].split()   
#             inp = [w for w in inp if len(w) > 0]
            q = x["Q"].split()
            ans= x["A"].split()
#             q = [w for w in q if len(w) > 0]
            
            inp_vector = [self.vocab.get(w) for w in inp]  # a list of word ids in the passage
    
            q_vector = [self.vocab.get(w) for w in q]

            ans_indices = [self.vocab.get(w) for w in ans]
            
            if (self.input_mask_mode == 'word'):
                input_mask = range(len(inp))
            elif (self.input_mask_mode == 'sentence'): # default is sentence
                input_mask = [index for index, w in enumerate(inp) if w == '.']  #input_mask store the position of sentence ends
            else:
                raise Exception("unknown input_mask_mode")
            fact_count = len(input_mask) # how  many sentences
    
            inputs.append(inp_vector)
            questions.append(q_vector)
            # NOTE: here we assume the answer is one word! 
            #note here answer is a word index
            answers.append(ans_indices)
#             answers.append(utils.process_word(word = x["A"], 
#                                             word2vec = self.word2vec, 
#                                             vocab = self.vocab, 
#                                             ivocab = self.ivocab, 
#                                             word_vector_size = self.word_vector_size, 
#                                             to_return = "index"))
            fact_counts.append(fact_count)
            input_masks.append(input_mask)
        
        return inputs, questions, answers, fact_counts, input_masks    

    def _test_rawinput_toid(self, data_raw):
        questions = []
        inputs = []
        answers = []
        fact_counts = []
        input_masks = []
        
        for x in data_raw:
            inp = x["C"].split()   # all lowercase
#             inp = [w for w in inp if len(w) > 0]
            q = x["Q"].split()
            ansSet= x["A"]#.lower().split()
#             q = [w for w in q if len(w) > 0]
            
            inp_vector = [self.vocab.get(w) for w in inp]  # a list of word embeddings in the passage
    
            q_vector = [self.vocab.get(w) for w in q]
            idlist_set=[]#set()
            for ansstr in ansSet:
                ans_word_list=ansstr.split()
                ans_indices = [self.vocab.get(w) for w in ans_word_list]
                idlist_set.append(ans_indices)
            
            if (self.input_mask_mode == 'word'):
                input_mask = range(len(inp))
            elif (self.input_mask_mode == 'sentence'): # default is sentence
                input_mask = [index for index, w in enumerate(inp) if w == '.']  #input_mask store the position of sentence ends
            else:
                raise Exception("unknown input_mask_mode")
            fact_count = len(input_mask) # how  many sentences
    
            inputs.append(inp_vector)
            questions.append(q_vector)
            # NOTE: here we assume the answer is one word! 
            #note here answer is a word index
            answers.append(idlist_set)
#             answers.append(utils.process_word(word = x["A"], 
#                                             word2vec = self.word2vec, 
#                                             vocab = self.vocab, 
#                                             ivocab = self.ivocab, 
#                                             word_vector_size = self.word_vector_size, 
#                                             to_return = "index"))
            fact_counts.append(fact_count)
            input_masks.append(input_mask)
        
        return inputs, questions, answers, fact_counts, input_masks 
        
    def _process_input(self, data_raw):
        questions = []
        inputs = []
        answers = []
        fact_counts = []
        input_masks = []
        
        for x in data_raw:
            inp = x["C"].lower().split()   # all lowercase
#             inp = [w for w in inp if len(w) > 0]
            q = x["Q"].lower().split()
#             q = [w for w in q if len(w) > 0]
            
            inp_vector = [utils.process_word(word = w, 
                                        word2vec = self.word2vec, 
                                        vocab = self.vocab, 
                                        ivocab = self.ivocab, 
                                        word_vector_size = self.word_vector_size, 
                                        to_return = "word2vec") for w in inp]  # a list of word embeddings in the passage
    
            q_vector = [utils.process_word(word = w, 
                                        word2vec = self.word2vec, 
                                        vocab = self.vocab, 
                                        ivocab = self.ivocab, 
                                        word_vector_size = self.word_vector_size, 
                                        to_return = "word2vec") for w in q]
            
            if (self.input_mask_mode == 'word'):
                input_mask = range(len(inp))
            elif (self.input_mask_mode == 'sentence'): # default is sentence
                input_mask = [index for index, w in enumerate(inp) if w == '.']  #input_mask store the position of sentence ends
            else:
                raise Exception("unknown input_mask_mode")
            fact_count = len(input_mask) # how  many sentences
    
            inputs.append(inp_vector)
            questions.append(q_vector)
            # NOTE: here we assume the answer is one word! 
            #note here answer is a word index
            answers.append(utils.process_word(word = x["A"], 
                                            word2vec = self.word2vec, 
                                            vocab = self.vocab, 
                                            ivocab = self.ivocab, 
                                            word_vector_size = self.word_vector_size, 
                                            to_return = "index"))
            fact_counts.append(fact_count)
            input_masks.append(input_mask)
        
        return inputs, questions, answers, fact_counts, input_masks 
    
    
    def get_batches_per_epoch(self, mode):
        if (mode == 'train'):
            return len(self.train_input) / self.batch_size
        elif (mode == 'test'):
            return len(self.test_input) / self.batch_size
        else:
            raise Exception("unknown mode")
    
    
    def shuffle_train_set(self):
        print "==> Shuffling the train set"
        combined = zip(self.train_input, self.train_q, self.train_answer, self.train_fact_count, self.train_input_mask)
        random.shuffle(combined)
        self.train_input, self.train_q, self.train_answer, self.train_fact_count, self.train_input_mask = zip(*combined)
    def load_word2vec_to_init(self, rand_values):
        
        for id, word in self.ivocab.iteritems():
            emb=self.word2vec.get(word)
            if emb is not None:
                rand_values[id]=np.array(emb)
        print '==> use word2vec initialization over...'
        return rand_values    
    
    def step(self, batch_index, mode):
        if mode == "train" and self.mode == "test":
            raise Exception("Cannot train during test mode")
        
        if mode == "train":
            theano_fn = self.train_fn 
            inputs = self.train_input
            qs = self.train_q
            answers = self.train_answer
            fact_counts = self.train_fact_count
            input_masks = self.train_input_mask
        if mode == "test":    
            theano_fn = self.test_fn 
            inputs = self.test_input
            qs = self.test_q
            answers = self.test_answer
            fact_counts = self.test_fact_count
            input_masks = self.test_input_mask
        
        start_index = batch_index * self.batch_size    
        inp = inputs[start_index:start_index+self.batch_size]
        q = qs[start_index:start_index+self.batch_size]
        ans = answers[start_index:start_index+self.batch_size]
        fact_count = fact_counts[start_index:start_index+self.batch_size]
        input_mask = input_masks[start_index:start_index+self.batch_size]
        
        param_norm = np.max([utils.get_norm(x.get_value()) for x in self.params])
        if mode == "train":
            inp, q, ans, fact_count, input_mask = self._process_batch(inp, q, ans, fact_count, input_mask, mode)
            ret = theano_fn(inp, q, ans, fact_count, input_mask)
            return {"prediction": ret[0],
                    "answers": ans,   #so, in training, ans is a list of paded list; in dev, ans is a list of set of list
                    "current_loss": ret[1],
                    "skipped": 0,
                    "log": "pn: %.3f" % param_norm,
                    }
        if mode == "test":
            #print 'start process_batch'    
            inp, q, fact_count, input_mask = self._process_batch(inp, q, ans, fact_count, input_mask, mode)
            #print 'start test a minibatch...'
            ret = theano_fn(inp, q, fact_count, input_mask)
            #print 'minibatch test finished, trying to return...'            
            return {"prediction": ret,
                    "answers": ans,   #so, in training, ans is a list of paded list; in dev, ans is a list of set of list
                    "current_loss": 0.0,
                    "skipped": 0,
                    "log": "pn: %.3f" % param_norm,
                    }        
        

        
