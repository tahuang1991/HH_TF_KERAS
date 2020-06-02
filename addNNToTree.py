import os
from array import array
import datetime
import keras
from math import sqrt
import numpy.lib.recfunctions as recfunctions
import numpy as np
import ROOT

import sys
import types

import sys
sys.argv.append( '-b' )
sys.argv.append( '-q' )

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

##local 
#sys.path.append("/Users/taohuang/Documents/DiHiggs/HH_Keras/HHTools/mvaTraining/")
import DNNModelLUT


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

__author__ = "Benjamin Peterson <benjamin@python.org>"
__version__ = "1.9.0"


# Useful for very coarse version differentiation.
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

if PY3:
    string_types = str,
    integer_types = int,
    class_types = type,
    text_type = str
    binary_type = bytes

    MAXSIZE = sys.maxsize
else:
    string_types = basestring,
    integer_types = (int, long)
    class_types = (type, types.ClassType)
    text_type = unicode
    binary_type = str

    if sys.platform.startswith("java"):
        # Jython always uses 32 bits.
        MAXSIZE = int((1 << 31) - 1)
    else:
        # It's possible to have sizeof(long) != sizeof(Py_ssize_t).
        class X(object):
            def __len__(self):
                return 1 << 31
        try:
            len(X())
        except OverflowError:
            # 32-bit
            MAXSIZE = int((1 << 31) - 1)
        else:
            # 64-bit
            MAXSIZE = int((1 << 63) - 1)
        del X


def _glob(filenames):
    """Glob a filename or list of filenames but always return the original
    string if the glob didn't match anything so URLs for remote file access
    are not clobbered.
    """
    if isinstance(filenames, string_types):
        filenames = [filenames]
    for name in filenames:
        matched_names = glob(name)
        if not matched_names:
            # use the original string
            matches.append(name)
        else:
            matches.extend(matched_names)
    return matches


def tree2array(tree,
               branches=None,
               selection=None,
               branchnames=None,
               object_selection=None,
               start=None,
               stop=None,
               step=None,
               include_weight=False,
               weight_name='weight',
               cache_size=-1):
    """Convert a tree into a numpy structured array.

    Convert branches of strings and basic types such as bool, int, float,
    double, etc. as well as variable-length and fixed-length multidimensional
    arrays and 1D or 2D vectors of basic types and strings. ``tree2array`` can
    also create columns in the output array that are expressions involving the
    TTree branches (i.e. ``'vect.Pt() / 1000'``) similar to ``TTree::Draw()``.
    See the notes below for important details.

    Parameters
    ----------
    tree : ROOT TTree instance
        The ROOT TTree to convert into an array.
    branches : list of strings and tuples or a string or tuple, optional (default=None)
        List of branches and expressions to include as columns of the array or
        a single branch or expression in which case a nonstructured array is
        returned. If None then include all branches that can be converted.
        Branches or expressions that result in variable-length subarrays can be
        truncated at a fixed length by using the tuple ``(branch_or_expression,
        fill_value, length)`` or converted into a single value with
        ``(branch_or_expression, fill_value)`` where ``length==1`` is implied.
        ``fill_value`` is used when the original array is shorter than
        ``length``. This truncation is after any object selection performed
        with the ``object_selection`` argument.
    selection : str, optional (default=None)
        Only include entries fulfilling this condition. If the condition
        evaluates to multiple values per tree entry (e.g. conditions on array
        branches) then an entry will be included if the condition evaluates to
        true for at least one array element.
    object_selection : dict, optional (default=None)
        A dictionary mapping selection strings to branch names or lists of
        branch names. Only array elements passing the selection strings will be
        included in the output array per entry in the tree. The branches
        specified must be variable-length array-type branches and the length of
        the selection and branches it acts on must match for each tree entry.
        For example ``object_selection={'a > 0': ['a', 'b']}`` will include all
        elements of 'a' and corresponding elements of 'b' where 'a > 0' for
        each tree entry. 'a' and 'b' must have the same length in every tree
        entry.
    start, stop, step: int, optional (default=None)
        The meaning of the ``start``, ``stop`` and ``step`` parameters is the
        same as for Python slices. If a range is supplied (by setting some of
        the ``start``, ``stop`` or ``step`` parameters), only the entries in
        that range and fulfilling the ``selection`` condition (if defined) are
        used.
    include_weight : bool, optional (default=False)
        Include a column containing the tree weight ``TTree::GetWeight()``.
        Note that this will be the same value for all entries unless the tree
        is actually a TChain containing multiple trees with different weights.
    weight_name : str, optional (default='weight')
        The field name for the weight column if ``include_weight=True``.
    cache_size : int, optional (default=-1)
        Set the size (in bytes) of the TTreeCache used while reading a TTree. A
        value of -1 uses ROOT's default cache size. A value of 0 disables the
        cache.

    Notes
    -----
    Types are converted according to the following table:

    .. _conversion_table:

    ========================  ===============================
    ROOT                      NumPy
    ========================  ===============================
    ``Bool_t``                ``np.bool``
    ``Char_t``                ``np.int8``
    ``UChar_t``               ``np.uint8``
    ``Short_t``               ``np.int16``
    ``UShort_t``              ``np.uint16``
    ``Int_t``                 ``np.int32``
    ``UInt_t``                ``np.uint32``
    ``Float_t``               ``np.float32``
    ``Double_t``              ``np.float64``
    ``Long64_t``              ``np.int64``
    ``ULong64_t``             ``np.uint64``
    ``<type>[2][3]...``       ``(<nptype>, (2, 3, ...))``
    ``<type>[nx][2]...``      ``np.object``
    ``string``                ``np.object``
    ``vector<t>``             ``np.object``
    ``vector<vector<t> >``    ``np.object``
    ========================  ===============================

    * Variable-length arrays (such as ``x[nx][2]``) and vectors (such as
      ``vector<int>``) are converted to NumPy arrays of the corresponding
      types.

    * Fixed-length arrays are converted to fixed-length NumPy array fields.

    **Branches with different lengths:**

    Note that when converting trees that have branches of different lengths
    into numpy arrays, the shorter branches will be extended to match the
    length of the longest branch by repeating their last values. If all
    requested branches are shorter than the longest branch in the tree, this
    will result in a "read failure" since beyond the end of the longest
    requested branch no additional bytes will be read from the file and
    root_numpy is unable to distinguish this from other ROOT errors that result
    in no bytes being read. In this case, explicitly set the ``stop`` argument
    to the length of the longest requested branch.


    See Also
    --------
    root2array
    array2root
    array2tree

    """
    #import ROOT
    import numpy as np
    if not isinstance(tree, ROOT.TTree):
        raise TypeError("tree must be a ROOT.TTree")

    if isinstance(branches, string_types):
        # single branch selected
        flatten = branches
        branches = [branches]
    elif isinstance(branches, tuple):
        if len(branches) not in (2, 3):
            raise ValueError(
                "invalid branch tuple: {0}. "
                "A branch tuple must contain two elements "
                "(branch_name, fill_value) or three elements "
                "(branch_name, fill_value, length) "
                "to yield a single value or truncate, respectively".format(branches))
        flatten = branches[0]
        branches = [branches]
    else:
        flatten = False

    if branchnames is None:
        branchnames = branches
    print "branches ",branches

    copytree = tree.CopyTree(selection)
    arr = copytree.AsMatrix(columns=branches )
    #arr = _librootnumpy.root2array_fromtree(
    #    cobj, branches, selection, object_selection,
    #    start, stop, step,
    #    include_weight,
    #    weight_name,
    #    cache_size)

    if flatten:
        # select single column
        return arr[flatten]
    dt = np.dtype({'names': branchnames, 'formats':[np.double]*len(branches)})
    arr.dtype = dt
    return arr




def array2tree(arr, branchnames, name='tree', tree=None):
    """Convert a numpy structured array into a ROOT TTree.

    Fields of basic types, strings, and fixed-size subarrays of basic types are
    supported. ``np.object`` and ``np.float16`` are currently not supported.

    Parameters
    ----------
    arr : array
        A numpy structured array
    name : str (optional, default='tree')
        Name of the created ROOT TTree if ``tree`` is None.
    tree : ROOT TTree (optional, default=None)
        An existing ROOT TTree to be extended by the numpy array. Any branch
        with the same name as a field in the numpy array will be extended as
        long as the types are compatible, otherwise a TypeError is raised. New
        branches will be created and filled for all new fields.

    Returns
    -------
    root_tree : a ROOT TTree

    Notes
    -----
    When using the ``tree`` argument to extend and/or add new branches to an
    existing tree, note that it is possible to create branches of different
    lengths. This will result in a warning from ROOT when root_numpy calls the
    tree's ``SetEntries()`` method. Beyond that, the tree should still be
    usable. While it might not be generally recommended to create branches with
    differing lengths, this behaviour could be required in certain situations.
    root_numpy makes no attempt to prevent such behaviour as this would be more
    strict than ROOT itself. Also see the note about converting trees that have
    branches of different lengths into numpy arrays in the documentation of
    :func:`tree2array`.

    See Also
    --------
    array2root
    root2array
    tree2array

    Examples
    --------

    Convert a numpy array into a tree:

    >>> from root_numpy import array2tree
    >>> import numpy as np
    >>>
    >>> a = np.array([(1, 2.5, 3.4),
    ...               (4, 5, 6.8)],
    ...              dtype=[('a', np.int32),
    ...                     ('b', np.float32),
    ...                     ('c', np.float64)])
    >>> tree = array2tree(a)
    >>> tree.Scan()
    ************************************************
    *    Row   *         a *         b *         c *
    ************************************************
    *        0 *         1 *       2.5 *       3.4 *
    *        1 *         4 *         5 *       6.8 *
    ************************************************

    Add new branches to an existing tree (continuing from the example above):

    >>> b = np.array([(4, 10),
    ...               (3, 5)],
    ...              dtype=[('d', np.int32),
    ...                     ('e', np.int32)])
    >>> array2tree(b, tree=tree)
    <ROOT.TTree object ("tree") at 0x1449970>
    >>> tree.Scan()
    ************************************************************************
    *    Row   *         a *         b *         c *         d *         e *
    ************************************************************************
    *        0 *         1 *       2.5 *       3.4 *         4 *        10 *
    *        1 *         4 *         5 *       6.8 *         3 *         5 *
    ************************************************************************

    """
    #import ROOT
    if tree is not None:
        if not isinstance(tree, ROOT.TTree):
            raise TypeError("tree must be a ROOT.TTree")
    else:
        tree  = ROOT.TTree(name, name)

    #if arr.dtype.names is None:
    #    raise NameError("array.dtype.names does not exist")


    #br_description = ":".join(name for name in arr.dtype.names)
    br_description = ":".join(name for name in branchnames)
    #for i in range(len(arr.dtype.names)):
    #    branchname = arr.dtype.names[i]
    #    dtype = arr.dtype.fields[branchname]
    #arr_str = "\n".join("\t".join( str(x) for x in row[0]) for row in arr)
    arr_str = "\n".join("\t".join( str(x) for x in row) for row in arr)
    istring        = ROOT.istringstream(arr_str)
    #print('arr-str',arr_str)
    tree.ReadStream(istring, br_description)
    tree.Scan()
    return tree


def plotmodel(modelLUT, model, plotname):
    modelfile = os.path.join(modelLUT[model]['workingdir'], "hh_resonant_trained_model.h5")
    thismodel = keras.models.load_model(modelfile)
    keras.utils.plot_model(thismodel, show_shapes = True, to_file=plotname)


def writeNNToTree(treename, input_file,  cut, modellist, modelLUT,  masslist, outFile):

    #file_handle = ROOT.TFile.Open(input_file)
    #tree = file_handle.Get(treename)
    ### convert all tree branches into array
    #a = tree2array(tree, branches=None, selection=cut) 
    df  =  ROOT.RDataFrame(treename, input_file)
    dataset = df.Filter(cut).Define('isSF_float','(1.0*isSF)').AsNumpy()

    #allinputs = map(lambda x : x if x != 'isSF' else 'isSF_float', dataset.keys())
    #alldataset = [np.ndarray(dataset[var]) for var in allinputs]
    allinputs = []
    alldataset = []
    for var in dataset.keys():
    #for var in ['jj_pt', 'll_pt', 'll_M', 'll_DR_l_l', 'jj_DR_j_j', 'llmetjj_DPhi_ll_jj', 'llmetjj_minDR_l_j', 'llmetjj_MTformula', 'mt2', 'isSF', 'jj_M', 'hme_h2mass_reco']:
        if var != 'isSF':
            #allinputs.append((var, 'f8'))
            allinputs.append(var)
            alldataset.append(np.array(dataset[var]))

    
    
    
    ## add mass column for parametric DNN
    allnnout = [] 
    nnoutname = []
    

    for i, key in enumerate(modellist):
	modelfile = os.path.join(modelLUT[key]['workingdir'], "hh_resonant_trained_model.h5")
	inputs = modelLUT[key]['inputs'] 
        parametricDNN = modelLUT[key]['parametricDNN'] 
        inputs = map(lambda x : x if x != 'isSF' else 'isSF_float', inputs)
        print "inputs ", inputs," model file ",modelfile
	thismodel = keras.models.load_model(modelfile)
        for mass in masslist:	
            #thisdataset = np.array(a[inputs].tolist(), dtype=np.float32)
            thisdataset = np.vstack([dataset[var] for var in inputs]).T
            print "thisdataset ",thisdataset.shape, " ", thisdataset[:2]
            branchname = "nnout_%s_M%d"%(key,mass)
            if parametricDNN:
                masscol = [mass]*len(thisdataset)
                thisdataset = np.c_[thisdataset, masscol]
	    nnout = thismodel.predict(thisdataset, batch_size=5000, verbose=1)
	    nnout = np.delete(nnout, 1, axis=1).flatten()
            print("nnout", nnout[:2])
	    allnnout.append(nnout)
	    nnoutname.append(branchname)

    
    

    ##for i in range(len(modellist)):
    for i in range(len(allnnout)):
        alldataset.append(allnnout[i])
        allinputs.append(nnoutname[i])
    #        recfunctions.append_fields(base, names, data=None, dtypes=None, fill_value=-1, usemask=True, asrecarray=False)
    #        a = recfunctions(a, nnoutname[i], allnnout[i])



    #dt = np.dtype({'names': allinputs, 'formats':[np.float]*len(allinputs)})

    print("all inputs ", allinputs)
    alldataset_arr = np.vstack(alldataset).T
    #print("alldataset ",alldataset_arr)
    #alldataset_arr.dtype = dt
    
    #a = np.array(alldataset_arr, dtype=dt)

    tfile = ROOT.TFile(outFile, "RECREATE")
    ch_new = array2tree(alldataset_arr, allinputs, "evtreeHME_nn")
    
    ch_new.GetCurrentFile().Write() 
    ch_new.GetCurrentFile().Close()

#
#output_folder = os.path.join(outdir, output_suffix)
output_folder = "./"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cut = '91-ll_M>15 && hme_h2mass_reco>=250'
#modellist = ['MTonly','MTandMT2','MTandMT2_MJJ', 'MTandMT2_HME']
#modellist = ['MTonly','MT2only','MTandMT2','MTandMT2_MJJ','MTandMT2_HME','MTandMT2_HMEMJJ', 'MTandMJJ']
modellist = ['MTandMT2_HMEMJJ']
#modellist = ['MTandMT2']
#modellist = ['MTonly']
#allfiles = os.listdir(filepath)
#allfiles = ['TT_all.root']
#treename = "evtreeHME_nn"
writeNNToTree(treename, filepath, cut, modellist, DNNModelLUT.ModelLUT,  [400], output_file)
def makeNtuple_prediction(masses):
     
    for shortname in full_local_samplelist.keys():    
        for samplename in full_local_samplelist[shortname].keys():
            f = full_local_samplelist[shortname][samplename]['path']
            output_file = os.path.join(output_folder, samplename+"_2020DNN.root")
            print "file ",f," output ", output_file 
            treename = "evtreeHME_nn"
            if "Radion" in samplename: treename = "Friends"
            writeNNToTree(treename, f, cut, modellist, DNNModelLUT.ModelLUT,  masses, output_file)

######## An example to add a branch to tree with DNN prediction:
#treename = "t"
#output_file = "test_M400_MTandMT2_HMEMJJ.root"
#filepath = "/Users/taohuang/Documents/DiHiggs/20180205_20180202_10k_Louvain_ALLNoSys/radion_M400_all.root"

#makeNtuple_prediction([260, 270, 300, 350, 400, 450, 500, 550, 600, 650, 750, 800, 900])
#makeNtuple_prediction([260, 400])
#for model in modellist:
#    plotmodel(SampleTypeLUT.ModelLUT, model, model+"_visualization.pdf")
