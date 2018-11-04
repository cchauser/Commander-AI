import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
from theano.tensor.opt import register_canonicalize
import time
import operator
import shelve
import itertools
import os
import math
import random
from copy import deepcopy


np.random.RandomState(1210542) #Mag seed

class LSTM(object):
    def __init__(self, model, numClasses, bType, hidden_dim = 64, bptt_truncate = 6):
        self.modelName = model
        self.numClasses = numClasses #0-10 inclusive
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.bType = bType
        
        
        
        try:
            print("Loading model...")
            db = shelve.open("{0}/{0}DB".format(model), "r")
            self.I = db['I']
            self.V = db['V']
            self.d = db['d']
            self.U_in = db['U_in']
            self.U_out = db['U_out']
            self.U_forget = db['U_forget']
            self.U_cand = db['U_cand']
            self.W_in = db['W_in']
            self.W_out = db['W_out']
            self.W_forget = db['W_forget']
            self.W_cand = db['W_cand']
            self.b_in = db['b_in']
            self.b_out = db['b_out']
            self.b_forget = db['b_forget']
            self.b_cand = db['b_cand']
            self.hidden_dim = self.I.get_value().shape[1]
            db.close()
            print("Model loaded")
        except:
            print("No model found")
            try:
                os.makedirs('{}'.format(model))
            except:
                pass
            db = shelve.open("{0}/{0}DB".format(model), "c")
            db.close()
    
            #Universal
            I = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (9, self.hidden_dim))
            V = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, numClasses))
            d = np.zeros(numClasses)
            
            #Layer 1
#            U_in = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
#            U_forget = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
#            U_out = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
#            U_cand = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
#            W_in = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
#            W_forget = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
#            W_out = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
#            W_cand = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            U_in = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (1, self.hidden_dim))
            U_forget = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (1, self.hidden_dim))
            U_out = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (1, self.hidden_dim))
            U_cand = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (1, self.hidden_dim))
            W_in = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (1, self.hidden_dim))
            W_forget = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (1, self.hidden_dim))
            W_out = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (1, self.hidden_dim))
            W_cand = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim), (1, self.hidden_dim))
            b_in = np.zeros((self.hidden_dim))
            b_forget = np.zeros((self.hidden_dim))
            b_out = np.zeros((self.hidden_dim))
            b_cand = np.zeros((self.hidden_dim))
            
            #Theano Shared Variables
            #Universal
            self.I = theano.shared(name='I', value=I.astype(theano.config.floatX))
            self.V = theano.shared(name='magV', value=V.astype(theano.config.floatX))
            self.d = theano.shared(name='dird', value=d.astype(theano.config.floatX))
            
            #Layer 1
            self.U_in = theano.shared(name='U_in', value=U_in.astype(theano.config.floatX))
            self.U_forget = theano.shared(name='U', value=U_forget.astype(theano.config.floatX))
            self.U_out = theano.shared(name='U_out', value=U_out.astype(theano.config.floatX))
            self.U_cand = theano.shared(name='U_cand', value=U_cand.astype(theano.config.floatX))
            self.W_in = theano.shared(name='W_in', value=W_in.astype(theano.config.floatX))
            self.W_forget = theano.shared(name='W_forget', value=W_forget.astype(theano.config.floatX))
            self.W_out = theano.shared(name='W_out', value=W_out.astype(theano.config.floatX))
            self.W_cand = theano.shared(name='W_cand', value=W_cand.astype(theano.config.floatX))
            self.b_in = theano.shared(name='b_in', value=b_in.astype(theano.config.floatX))
            self.b_forget = theano.shared(name='b_forget', value=b_forget.astype(theano.config.floatX))
            self.b_out = theano.shared(name='b_out', value=b_out.astype(theano.config.floatX))
            self.b_cand = theano.shared(name='b_cand', value=b_cand.astype(theano.config.floatX))

            
        finally:
            #Derivatives
            #Universal
            self.mI = theano.shared(name='mI', value=np.zeros(self.I.get_value().shape).astype(theano.config.floatX))
            self.vI = theano.shared(name='vI', value=np.zeros(self.I.get_value().shape).astype(theano.config.floatX))
            self.mV = theano.shared(name='mmagV', value=np.zeros(self.V.get_value().shape).astype(theano.config.floatX))
            self.vV = theano.shared(name='vmagV', value=np.zeros(self.V.get_value().shape).astype(theano.config.floatX))
            self.md = theano.shared(name='mdird', value=np.zeros(self.d.get_value().shape).astype(theano.config.floatX))
            self.vd = theano.shared(name='vmagd', value=np.zeros(self.d.get_value().shape).astype(theano.config.floatX))
            
            #Layer 1
            self.mU_in = theano.shared(name='mU_in', value=np.zeros(self.U_in.get_value().shape).astype(theano.config.floatX))
            self.mU_forget = theano.shared(name='mU_forget', value=np.zeros(self.U_forget.get_value().shape).astype(theano.config.floatX))
            self.mU_out = theano.shared(name='mU_out', value=np.zeros(self.U_out.get_value().shape).astype(theano.config.floatX))
            self.mU_cand = theano.shared(name='mU_cand', value=np.zeros(self.U_cand.get_value().shape).astype(theano.config.floatX))
            self.mW_in = theano.shared(name='mW_in', value=np.zeros(self.W_in.get_value().shape).astype(theano.config.floatX))
            self.mW_forget = theano.shared(name='mW_forget', value=np.zeros(self.W_forget.get_value().shape).astype(theano.config.floatX))
            self.mW_out = theano.shared(name='mW_out', value=np.zeros(self.W_out.get_value().shape).astype(theano.config.floatX))
            self.mW_cand = theano.shared(name='mW_cand', value=np.zeros(self.W_cand.get_value().shape).astype(theano.config.floatX))
            self.mb_in = theano.shared(name='mb_in', value=np.zeros(self.b_in.get_value().shape).astype(theano.config.floatX))
            self.mb_forget = theano.shared(name='mb_forget', value=np.zeros(self.b_forget.get_value().shape).astype(theano.config.floatX))
            self.mb_out = theano.shared(name='mb_out', value=np.zeros(self.b_out.get_value().shape).astype(theano.config.floatX))
            self.mb_cand = theano.shared(name='mb_cand', value=np.zeros(self.b_cand.get_value().shape).astype(theano.config.floatX))
            self.vU_in = theano.shared(name='vU_in', value=np.zeros(self.U_in.get_value().shape).astype(theano.config.floatX))
            self.vU_forget = theano.shared(name='vU_forget', value=np.zeros(self.U_forget.get_value().shape).astype(theano.config.floatX))
            self.vU_out = theano.shared(name='vU_out', value=np.zeros(self.U_out.get_value().shape).astype(theano.config.floatX))
            self.vU_cand = theano.shared(name='vU_cand', value=np.zeros(self.U_cand.get_value().shape).astype(theano.config.floatX))
            self.vW_in = theano.shared(name='vW_in', value=np.zeros(self.W_in.get_value().shape).astype(theano.config.floatX))
            self.vW_forget = theano.shared(name='vW_forget', value=np.zeros(self.W_forget.get_value().shape).astype(theano.config.floatX))
            self.vW_out = theano.shared(name='vW_out', value=np.zeros(self.W_out.get_value().shape).astype(theano.config.floatX))
            self.vW_cand = theano.shared(name='vW_cand', value=np.zeros(self.W_cand.get_value().shape).astype(theano.config.floatX))
            self.vb_in = theano.shared(name='vb_in', value=np.zeros(self.b_in.get_value().shape).astype(theano.config.floatX))
            self.vb_forget = theano.shared(name='vb_forget', value=np.zeros(self.b_forget.get_value().shape).astype(theano.config.floatX))
            self.vb_out = theano.shared(name='vb_out', value=np.zeros(self.b_out.get_value().shape).astype(theano.config.floatX))
            self.vb_cand = theano.shared(name='vb_cand', value=np.zeros(self.b_cand.get_value().shape).astype(theano.config.floatX))


        self.theano = {}
        print("Loading functions...")
#        try:
        self.__theano_build__()
#            print("Done\nReady\n")
#        except:
#            print("Error loading functions")

    def _step(self, x, o_Prev, C_Prev):
        #Transform input from (1,9) to (1,hidden_dim)
        x_ = T.dot(x, self.I)
        
        #Gate calculations; these modulate how much of the previous layer's data is used
        #The sigmoid function squishes the values to between 0 and 1 (effectively a percentage value)
#        f_g = T.nnet.sigmoid(T.dot(self.W_forget, x_) + T.dot(self.U_forget, o_Prev) + self.b_forget) #256,1
#        i_g = T.nnet.sigmoid(T.dot(self.W_in, x_) + T.dot(self.U_in, o_Prev) + self.b_in)
#        o_g = T.nnet.sigmoid(T.dot(self.W_out, x_) + T.dot(self.U_out, o_Prev) + self.b_out)
        f_g = T.nnet.sigmoid(self.W_forget * x_ + self.U_forget * o_Prev + self.b_forget) #256,1
        i_g = T.nnet.sigmoid(self.W_in * x_ + self.U_in * o_Prev + self.b_in)
        o_g = T.nnet.sigmoid(self.W_out * x_ + self.U_out * o_Prev + self.b_out)


        #Candidate value for the cell memory that runs along the neural network
#        C_c = T.nnet.nnet.softsign(T.dot(self.W_cand, x_) + T.dot(self.U_cand, o_Prev) +  self.b_cand) #Softsign activation
#        C_c = T.dot(self.W_cand, x_) + T.dot(self.U_cand, o_Prev) +  self.b_cand #Linear activation
        C_c = self.W_cand * x_ + self.U_cand * o_Prev +  self.b_cand #Linear activation
#        C_c = T.switch(C_c < 0, C_c * .01, C_c) #Leaky ReLU activation
        
        #New cell value actual
        C = i_g * C_c + f_g * C_Prev

        #Cell output
#        o = T.nnet.nnet.softsign(o_g * C) * 5 #Softsign output activation. Multiplied by 5 so sigmoid can reach extremes
        o = o_g * C #Linear output activation

        return o, C
        
    def __theano_build__(self):

        x = T.matrix('x') #Input sequence stored as theano variable x
        y = T.scalar('y') #Target magnitude output value stored as theano variable magy
        learnRate = T.scalar('learnRate')
        t = T.scalar('t') #Time step
        
        print("Loading _step")
        [o, C], updates = theano.scan(self._step,
                                      sequences=x,
                                      truncate_gradient = self.bptt_truncate,
                                      outputs_info=[theano.shared(value = np.zeros((1,self.hidden_dim)).astype(theano.config.floatX)),
                                                    theano.shared(value = np.ones((1,self.hidden_dim)).astype(theano.config.floatX))])

        gameState = o[-1] # The gamestate as the nerual network sees it

        self.debug_gameState = theano.function([x], gameState)
        
        pred_Prob = T.nnet.sigmoid(T.dot(gameState, self.V) + self.d) 
        
        #Calculate nerual network output layer and error
        if self.bType == 'dir':
            pred_error = T.sum((((y - pred_Prob) + .5) % 1 - .5)**2)
#            pred_error = T.sum((y-pred_Prob)**2)
        else:
#            pred_Prob = T.nnet.softmax(T.dot(gameState, self.V) + self.d)[0]#returns a 2d matrix with one row. so just take that row
            pred_error = T.sum((pred_Prob - y)**2)

        move = pred_Prob
                               
        print("Loading f_pred_Prob")
        #Declare theano functions for predicting outcomes
        self.f_pred_Prob = theano.function([x], [pred_Prob]) #Returns the probability vector
        print("Loading f_pred")
        
        self.get_move = theano.function([x], move) #Gets move for the AI

        #Define function for calculating error
        print("Loading ce_error")
        self.ce_error = theano.function([x, y], pred_error, allow_input_downcast=True) #Returns cross-entropy error

        print("Loading gradients")
        
      ###Gradients###
        print("-Loading derivatives")
        #Universal
        dI = T.grad(pred_error, self.I)
        dV = T.grad(pred_error, self.V)
        dd = T.grad(pred_error, self.d)

        #Layer 1
        print("--Layer1")
        dW_in = T.grad(pred_error, self.W_in)
        dW_out = T.grad(pred_error, self.W_out)
        dW_forget = T.grad(pred_error, self.W_forget)
        dW_cand = T.grad(pred_error, self.W_cand)
        dU_in = T.grad(pred_error, self.U_in)
        dU_out = T.grad(pred_error, self.U_out)
        dU_forget = T.grad(pred_error, self.U_forget)
        dU_cand = T.grad(pred_error, self.U_cand)
        db_in = T.grad(pred_error, self.b_in)
        db_out = T.grad(pred_error, self.b_out)
        db_forget = T.grad(pred_error, self.b_forget)
        db_cand = T.grad(pred_error, self.b_cand)

     ###Adam cache updates###
        beta1 = .9
        beta2 = .999
        eps = 1e-8
        print("-Loading cache updates")
        #Universal
        mI = beta1 * self.mI + (1 - beta1) * dI
        vI = beta1 * self.vI + (1 - beta1) * (dI ** 2)
        mV = beta1 * self.mV + (1 - beta1) * dV
        vV = beta1 * self.vV + (1 - beta1) * (dV ** 2)
        md = beta1 * self.md + (1 - beta1) * dd
        vd = beta2 * self.vd + (1 - beta2) * (dd ** 2)

        print("--Layer1")
        #Layer 1
        mW_in = beta1 * self.mW_in + (1 - beta1) * dW_in
        mW_out = beta1 * self.mW_out + (1 - beta1) * dW_out
        mW_forget = beta1 * self.mW_forget + (1 - beta1) * dW_forget
        mW_cand = beta1 * self.mW_cand + (1 - beta1) * dW_cand
        mU_in = beta1 * self.mU_in + (1 - beta1) * dU_in
        mU_out = beta1 * self.mU_out + (1 - beta1) * dU_out
        mU_forget = beta1 * self.mU_forget + (1 - beta1) * dU_forget
        mU_cand = beta1 * self.mU_cand + (1 - beta1) * dU_cand
        mb_in = beta1 * self.mb_in + (1 - beta1) * db_in
        mb_out = beta1 * self.mb_out + (1 - beta1) * db_out
        mb_forget = beta1 * self.mb_forget + (1 - beta1) * db_forget
        mb_cand = beta1 * self.mb_cand + (1 - beta1) * db_cand
        vW_in = beta2 * self.vW_in + (1 - beta2) * dW_in ** 2
        vW_out = beta2 * self.vW_out + (1 - beta2) * dW_out ** 2
        vW_forget = beta2 * self.vW_forget + (1 - beta2) * dW_forget ** 2
        vW_cand = beta2 * self.vW_cand + (1 - beta2) * dW_cand ** 2
        vU_in = beta2 * self.vU_in + (1 - beta2) * dU_in ** 2
        vU_out = beta2 * self.vU_out + (1 - beta2) * dU_out ** 2
        vU_forget = beta2 * self.vU_forget + (1 - beta2) * dU_forget ** 2
        vU_cand = beta2 * self.vU_cand + (1 - beta2) * dU_cand ** 2
        vb_in = beta2 * self.vb_in + (1 - beta2) * db_in ** 2
        vb_out = beta2 * self.vb_out + (1 - beta2) * db_out ** 2
        vb_forget = beta2 * self.vb_forget + (1 - beta2) * db_forget ** 2
        vb_cand = beta2 * self.vb_cand + (1 - beta2) * db_cand ** 2

        #If it's ugly but it works then it's not ugly
        print("Loading adam_step")        
        self.adam_step = theano.function(
            [x, y, learnRate, t],
            [], 
            updates=[(self.I, self.I - learnRate * (mI / (1-(beta1 ** t))) / (T.sqrt((vI / (1-(beta2 ** t)))) + eps)),
                     (self.V, self.V - learnRate * (mV / (1-(beta1 ** t))) / (T.sqrt((vV / (1-(beta2 ** t)))) + eps)),
                     (self.d, self.d - learnRate * (md / (1-(beta1 ** t))) / (T.sqrt((vd / (1-(beta2 ** t)))) + eps)),
                     (self.W_in, self.W_in - learnRate * (mW_in / (1-(beta1 ** t))) / (T.sqrt((vW_in / (1-(beta2 ** t)))) + eps)),
                     (self.W_out, self.W_out - learnRate * (mW_out / (1-(beta1 ** t))) / (T.sqrt((vW_out / (1-(beta2 ** t)))) + eps)),
                     (self.W_forget, self.W_forget - learnRate * (mW_forget / (1-(beta1 ** t))) / (T.sqrt((vW_forget / (1-(beta2 ** t)))) + eps)),
                     (self.W_cand, self.W_cand - learnRate * (mW_cand / (1-(beta1 ** t))) / (T.sqrt((vW_cand / (1-(beta2 ** t)))) + eps)),
                     (self.U_in, self.U_in - learnRate * (mU_in / (1-(beta1 ** t))) / (T.sqrt((vU_in / (1-(beta2 ** t)))) + eps)),
                     (self.U_out, self.U_out - learnRate * (mU_out / (1-(beta1 ** t))) / (T.sqrt((vU_out / (1-(beta2 ** t)))) + eps)),
                     (self.U_forget, self.U_forget - learnRate * (mU_forget / (1-(beta1 ** t))) / (T.sqrt((vU_forget / (1-(beta2 ** t)))) + eps)),
                     (self.U_cand, self.U_cand - learnRate * (mU_cand / (1-(beta1 ** t))) / (T.sqrt((vU_cand / (1-(beta2 ** t)))) + eps)),
                     (self.b_in, self.b_in - learnRate * (mb_in / (1-(beta1 ** t))) / (T.sqrt((vb_in / (1-(beta2 ** t)))) + eps)),
                     (self.b_out, self.b_out - learnRate * (mb_out / (1-(beta1 ** t))) / (T.sqrt((vb_out / (1-(beta2 ** t)))) + eps)),
                     (self.b_forget, self.b_forget - learnRate * (mb_forget / (1-(beta1 ** t))) / (T.sqrt((vb_forget / (1-(beta2 ** t)))) + eps)),
                     (self.b_cand, self.b_cand - learnRate * (mb_cand / (1-(beta1 ** t))) / (T.sqrt((vb_cand / (1-(beta2 ** t)))) + eps)),
                     (self.mI, mI),
                     (self.vI, vI),
                     (self.mV, mV),
                     (self.vV, vV),
                     (self.md, md),
                     (self.vd, vd),
                     (self.mW_in, mW_in),
                     (self.mW_out, mW_out),
                     (self.mW_forget, mW_forget),
                     (self.mW_cand, mW_cand),                 
                     (self.mU_in, mU_in),
                     (self.mU_out, mU_out),
                     (self.mU_forget, mU_forget),
                     (self.mU_cand, mU_cand),                     
                     (self.mb_in, mb_in),
                     (self.mb_out, mb_out),
                     (self.mb_forget, mb_forget),
                     (self.mb_cand, mb_cand),
                     (self.vW_in, vW_in),
                     (self.vW_out, vW_out),
                     (self.vW_forget, vW_forget),
                     (self.vW_cand, vW_cand),  
                     (self.vU_in, vU_in),
                     (self.vU_out, vU_out),
                     (self.vU_forget, vU_forget),
                     (self.vU_cand, vU_cand),                     
                     (self.vb_in, vb_in),
                     (self.vb_out, vb_out),
                     (self.vb_forget, vb_forget),
                     (self.vb_cand, vb_cand)
                     ])

    def mClass(self, y):
        return y / 10

    def dClass(self, y):
        return y / 360

    def learning_step(self, x, y, learnRate, t, heading = None, returnError = False):
        if self.bType == 'mag':
            y_ = self.mClass(y)
        elif self.bType == 'dir':
            y_ = self.dClass(y)
        else:
            raise Exception("UNKNOWN TYPE")
            
        x_ = np.asarray(x)
        if heading:
            x_[0][6] = heading
        
        self.adam_step(x_, y_, learnRate, t)
        
        move = self.get_move(x_)[0][0]
        if not returnError:
            if self.bType == 'dir':
                return move * 360
            else:
                return move * 10
        else:
            return np.sqrt(self.ce_error(x_, y_))
    
    def get_error(self, x, y, heading = None):
        if self.bType == 'mag':
            y_ = self.mClass(y)
        elif self.bType == 'dir':
            y_ = self.dClass(y)
        else:
            raise Exception("UNKNOWN TYPE")
            
        x_ = np.asarray(x)
        if heading:
            x_[0][6] = heading
        
        return np.sqrt(self.ce_error(x_, y_))

    def nnet_move(self, packet, heading = None):
        if heading and self.bType == 'mag':
            packet[0][6] = heading
        packet_ = np.asarray(packet)
        move = self.get_move(packet_)[0][0]
        if self.bType == 'dir':
            return move * 360
        else:
            return move * 10

    def save_model(self):
        try:
            db = shelve.open("{0}/{0}DB".format(self.modelName), "r")
            db.close()
        except:
            os.makedirs('{}'.format(self.test_name))
            db = shelve.open("{0}/{0}DB".format(self.modelName), "c")
            db.close()
        finally:
            db = shelve.open("{0}/{0}DB".format(self.modelName), "w")
            db['I'] = self.I
            db['V'] = self.V
            db['d'] = self.d
            db['U_in'] = self.U_in
            db['U_out'] = self.U_out
            db['U_forget'] = self.U_forget
            db['U_cand'] = self.U_cand
            db['W_in'] = self.W_in
            db['W_out'] = self.W_out
            db['W_forget'] = self.W_forget
            db['W_cand'] = self.W_cand
            db['b_in'] = self.b_in
            db['b_out'] = self.b_out
            db['b_forget'] = self.b_forget
            db['b_cand'] = self.b_cand
            db.close()
