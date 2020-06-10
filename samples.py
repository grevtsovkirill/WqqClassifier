###
BASE       = '../Files/processed/'

####
samples = {    
    'ttW' : {
        'filename' : ['skimReco_700000_a.csv','skimReco_700000_d.csv','skimReco_700000_e.csv'],
        #'filename' : ['skimReco_700000_a.csv'],
        'color'    : 'tomato',
        'group'   : 'ttW',
        'class'   : 1.
    },
    'ttZmumu' : {
        'filename' : 'skimReco_410219_xs.csv',   
        'group'   : 'ttZ',   
        'color'    : 'steelblue',
        'class'   : -99.
        
    },    
    'ttbar' : {
        'filename' : ['skimReco_410470_a.csv','skimReco_410470_d.csv','skimReco_410470_e.csv'],   
        #'filename' : ['skimReco_410470_a.csv'],
        #'filename' : ['skimReco_700000_a.csv'],
        'color'    : 'slategray',
        'group'   : 'ttbar',
        'class'   : 0.
    }
}
