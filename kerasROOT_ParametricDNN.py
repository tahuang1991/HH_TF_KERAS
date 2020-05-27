import ROOT

# Select Theano as backend for Keras
from os import environ
environ['KERAS_BACKEND'] = 'theano'

# Set architecture of system (AVX instruction set is not supported on SWAN)
#environ['THEANO_FLAGS'] = 'gcc.cxxflags=-march=corei7'

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam



def get_xsection_eventweightsum_tree(tree):
    #tree = ROOT.TChain( treename )
    #tree.Add(filename)
    n = tree.GetEntry()
    i = 0
    xsec = 0.0; event_weight_sum =0.0;
    for i in range(0, 100):
        tree.GetEntry(i)
        cross_section = tree.cross_section
        weisum = tree.event_weight_sum
        if i == 0:
            xsec = cross_section
            event_weight_sum = weisum
        else:
            if abs(xsec-cross_section)>0.01*xsec or abs(event_weight_sum - weisum)> 0.01*event_weight_sum:
                print "WARNING: cross_section or event_weight_sum may be not a single value, xsec  ", xsec," event_weight_sum ",event_weight_sum," this entry ",cross_section," ",weisum 
    return xsec,event_weight_sum


### get signal

