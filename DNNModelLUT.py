


ModelLUT = { 
	'MTonly':{
	    'workingdir' : '/Users/taohuang/Documents/DiHiggs/HH_TF_KERAS_2020/hh_resonant_trained_models_kinematicwithMTonly_20180206/2018-02-06_100epochs/',
            'legend': 'Kinematic + MT',#2018
	    'inputs': ['jj_pt', 'll_pt', 'll_M', 'll_DR_l_l', 'jj_DR_j_j', 'llmetjj_DPhi_ll_jj', 'llmetjj_minDR_l_j', 'llmetjj_MTformula', 'isSF'],
            'parametricDNN' : True,
		},
	'MTandMT2':{
	    'workingdir' : '/Users/taohuang/Documents/DiHiggs/HH_TF_KERAS_2020/hh_resonant_trained_models_kinematicwithMTandMT2_20180206/2018-02-07_100epochs/',
            'legend': 'Kinematic + MT + MT2',#2018
	    'inputs': ['jj_pt', 'll_pt', 'll_M', 'll_DR_l_l', 'jj_DR_j_j', 'llmetjj_DPhi_ll_jj', 'llmetjj_minDR_l_j', 'llmetjj_MTformula','mt2','isSF'],
            'parametricDNN' : True,
		},
	'MTandMT2_MJJ':{ 
	    'workingdir' : '/Users/taohuang/Documents/DiHiggs/HH_TF_KERAS_2020/hh_resonant_trained_models_kinematicwithMTandMT2Mjj_20180206/2018-02-07_100epochs/',
	    'legend': 'Kinematic+MT+MT2+Mjj',#2018
	    'inputs': ['jj_pt', 'll_pt', 'll_M', 'll_DR_l_l', 'jj_DR_j_j', 'llmetjj_DPhi_ll_jj', 'llmetjj_minDR_l_j', 'llmetjj_MTformula', 'mt2', 'isSF', 'jj_M'],
            'parametricDNN' : True,
	    },
	'MTandMT2_HMEMJJ':{ 
	    'workingdir' : '/Users/taohuang/Documents/DiHiggs/HH_TF_KERAS_2020/hh_resonant_trained_models_kinematicwithMTandMT2MjjHME_20180206/2018-02-07_100epochs/',
	    'legend': 'Kinematic+MT+MT2+Mjj+HME',#2018
	    'inputs': ['jj_pt', 'll_pt', 'll_M', 'll_DR_l_l', 'jj_DR_j_j', 'llmetjj_DPhi_ll_jj', 'llmetjj_minDR_l_j', 'llmetjj_MTformula', 'mt2', 'isSF', 'jj_M', 'hme_h2mass_reco'],
            'parametricDNN' : True,
	    }

	}

