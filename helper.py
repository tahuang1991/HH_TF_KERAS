import ROOT

_rootBranchType2PythonArray = { 'b':'B', 'B':'b', 'i':'I', 'I':'i', 'F':'f', 'D':'d', 'l':'L', 'L':'l', 'O':'B' }

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

def get_xsection_eventweightsum_file(filename, treename):
    tree = ROOT.TChain( treename )
    tree.Add(filename)
    return get_xsection_eventweightsum_tree(tree)

def add_parametric_variable_tree(tree, branchname, column):

    rootBranchType = "F"
    variable =  array(_rootBranchType2PythonArray[rootBranchType], [0.0])
    branch = tree.Branch( branchname, variable ,"%s/%s" % (branchname, rootBranchType))

    tree.SetBranchStatus("*", 0)
    entries = tree.GetEntries()
    for i in range(entries):
        tree.GetEntry(i)
        variable[0] = column[i]
        tree.Fill()





