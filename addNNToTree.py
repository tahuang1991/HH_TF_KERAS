import os
from array import array
import datetime
import keras
from math import sqrt
import numpy.lib.recfunctions as recfunctions
import numpy as np
import ROOT

import sys
sys.argv.append( '-b' )
sys.argv.append( '-q' )

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

##local 
#sys.path.append("/Users/taohuang/Documents/DiHiggs/HH_Keras/HHTools/mvaTraining/")
from common import *
import SampleTypeLUT


ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.Reset()


#ROOT.gROOT.SetBatch(1)
#ROOT.gStyle.SetStatW(0.07)
#ROOT.gStyle.SetStatH(0.06)


#ROOT.gStyle.SetErrorX(0)
#ROOT.gStyle.SetErrorY(0)

#ROOT.gStyle.SetTitleStyle(0)
#ROOT.gStyle.SetTitleAlign(13) ## coord in top left
#ROOT.gStyle.SetTitleX(0.)
#ROOT.gStyle.SetTitleY(1.)
#ROOT.gStyle.SetTitleW(1)
#ROOT.gStyle.SetTitleH(0.058)
#ROOT.gStyle.SetTitleBorderSize(0)

#ROOT.gStyle.SetPadLeftMargin(0.126)
#ROOT.gStyle.SetPadRightMargin(0.10)
#ROOT.gStyle.SetPadTopMargin(0.06)
#ROOT.gStyle.SetPadBottomMargin(0.13)

#ROOT.gStyle.SetMarkerStyle(1)
### expected entries and new weight should be added if some subjobs are failed
#expectedEntries = {'DYToLL1J': 5095747, 'radion_M600': 25970, 'sT_top': 27069, 'radion_M260': 9679, 'radion_M750': 29187, 'TT': 5487534, 'radion_M270': 10015, 'radion_M400': 18036, 'radion_M650': 27518, 'radion_M500': 22384, 'radion_M350': 15166, 'DYM10to50': 69311, 'radion_M450': 20295, 'radion_M800': 30349, 'radion_M900': 31338, 'radion_M300': 12251, 'DYToLL2J': 15265923, 'sT_antitop': 27460, 'DYToLL0J': 838304, 'radion_M550': 24578}

def plotmodel(modelLUT, model, plotname):
    modelfile = os.path.join(modelLUT[model]['workingdir'], "hh_resonant_trained_model.h5")
    thismodel = keras.models.load_model(modelfile)
    keras.utils.plot_model(thismodel, show_shapes = True, to_file=plotname)


def writeNNToTree(input_file, weight, modellist, modelLUT,  masslist, outFile):
    cut = "(91 - ll_M) > 15.0"
    FullInputs = ['jj_pt', 'll_pt', 'll_M', 'll_DR_l_l', 'jj_DR_j_j', 'llmetjj_DPhi_ll_jj', 'llmetjj_minDR_l_j', 'llmetjj_MTformula','mt2', 'jj_M','isSF','hme_h2mass_reco','isElEl','isElMu','isMuEl','isMuMu','met_pt','cross_section','event_weight_sum','sample_weight','event_reco_weight']
    #for lep in ['lep1','lep2']:
    #    for sys in ['Isosf','IDsf','trgsf','trackingsf']:
    #        for plottype in ['up','down']:
    #            FullInputs.append(lep+sys+"_"+plottype)
    #for sys in ["jjbtag_heavy","jjbtag_light","event_pu_weight","event_pdf_weight"]:
    #    for plottype in ["up","down"]:
    #        FullInputs.append(sys+"_"+plottype)

    #'cross_section','event_weight_sum'
    #FIXME add cross_section, event_weight_sum
    #dataset, weight = tree_to_numpy(f, FullInputs, weight_expression, cut, reweight_to_cross_section=False)
    file_handle = TFile.Open(input_file)

    #tree = file_handle.Get('t')
    tree = file_handle.Get('Friends')

    print "FullInputs ",FullInputs
    # Read the tree and convert it to a numpy structured array
    a = tree2array(tree, branches= FullInputs + [weight], selection=cut)
    # Rename the last column to 'weight'
    #print "print a type ", type(a)," shape ",a.shape, " len(a) ",len(a), " a.dtype.names ",a.dtype.names, " a[:10,:] ", a[:10]
    a.dtype.names = FullInputs + ['final_total_weight']

  
    #print "print a type ", type(a)," shape ",a.shape, " a.dtype.names ",a.dtype.names, " a[:10,:] ", a[:10]
    
    
    ## add mass column
    allnnout = [] 
    nnoutname = []
    
    #sample_type = ROOT.vector('string')()
    for i, key in enumerate(modellist):
	modelfile = os.path.join(modelLUT[key]['workingdir'], "hh_resonant_trained_model.h5")
	inputs = modelLUT[key]['inputs'] 
        print "inputs ", inputs," model file ",modelfile
	thismodel = keras.models.load_model(modelfile)
        for mass in masslist:	
	    thisdataset = np.array(a[inputs].tolist(), dtype=np.float32)
            print "thisdataset ",thisdataset.shape, " ", thisdataset[:2]
	    masscol = [mass]*len(a)
	    thisdataset = np.c_[thisdataset, masscol]
	    nnout = thismodel.predict(thisdataset, batch_size=5000, verbose=1)
	    nnout = np.delete(nnout, 1, axis=1).flatten()
	    allnnout.append(nnout)
	    nnoutname.append("nnout_%s_M%d"%(key,mass))


    
    #for i in range(len(modellist)):
    for i in range(len(allnnout)):
	    #recfunctions.append_fields(base, names, data=None, dtypes=None, fill_value=-1, usemask=True, asrecarray=False)
	    a = recfunctions.append_fields(a, nnoutname[i], allnnout[i])
	    #a = recfunctions.(a, nnoutname[i], allnnout[i])

    #dataset.dtype.names = FullInputs + ['weight', 'mass'] + nnoutname
    #print "print a type ", type(a)," shape ",a.shape, " a.dtype.names ",a.dtype.names, " a[:10,:] ", a[:10]
    print "\n a dytpenames ", a.dtype.names 

    tfile = ROOT.TFile(outFile, "RECREATE")
    #ch_new = array2tree(dataset,  FullInputs + ['mass'] + nnoutname + ['finalweight'], "evtreeHME_nn")
    ch_new = array2tree(a, "evtreeHME_nn")
    
    ch_new.GetCurrentFile().Write() 
    ch_new.GetCurrentFile().Close()


#filepath = "/Users/taohuang/Documents/DiHiggs/20180316_NanoAOD/20180619_HHbbWW_addHME_addSys_10k/"
#outdir = "/Users/taohuang/Documents/DiHiggs/20170530/20171021_Louvain_addNN/"
outdir = filepath
suffix = 'addNN'
date_suffix = '{:%Y-%m-%d}'.format(datetime.date.today())
output_suffix = '2018-06-19_addNN'
output_folder = os.path.join(outdir, output_suffix)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#modellist = ['MTonly','MTandMT2','MTandMT2_MJJ', 'MTandMT2_HME']
#modellist = ['MTonly','MT2only','MTandMT2','MTandMT2_MJJ','MTandMT2_HME','MTandMT2_HMEMJJ', 'MTandMJJ']
#modellist = ['MTandMT2']
modellist = ['MTonly']
allfiles = os.listdir(filepath)
#allfiles = ['TT_all.root']
def makeNtuple_prediction(masses):
     
    for f in allfiles:
	if f.endswith(".root"):
            #name = f.split('.')[0][:-4]
            ##filename: TTTo2L2Nu_TuneCUETP8M2_ttHtranche3_13TeV-powheg-pythia8_HME_Friends.root
            name = f.split('.')[0]
            print "file ",f," samplename ",name
	    #addsampleType(filepath+f, sampletype, name+"_addType.root")
	    #weight = "event_weight * trigeff * jjbtag_heavy * jjbtag_light * llidiso * pu"
            #weight = "total_weight"
            weight = "sample_weight*event_reco_weight"
            #if name.startswith("DY"):
            #	weight = weight+"* dy_nobtag_to_btagM_weight"
	    print 'file ',file,' name ',name," weight ",weight
            outname = name+"_NN.root"
            output_file = os.path.join(output_folder, outname)		
	    if "Radion" in name:
                mass = int( name.split("_")[1][2:] )
	        writeNNToTree(filepath+f, weight, modellist, SampleTypeLUT.ModelLUT,  [mass], output_file)
	    else:
	        writeNNToTree(filepath+f, weight, modellist, SampleTypeLUT.ModelLUT,  masses, output_file)


#makeNtuple_prediction([260, 270, 300, 350, 400, 450, 500, 550, 600, 650, 750, 800, 900])
makeNtuple_prediction([260, 400])
#for model in modellist:
#    plotmodel(SampleTypeLUT.ModelLUT, model, model+"_visualization.pdf")
