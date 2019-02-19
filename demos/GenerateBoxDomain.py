### This won't be needed once the package is part of local python path ###
import sys
sys.path.insert(0, '../dev/')



import WindSE

### Create an Instance of the Options ###
options = WindSE.Options()

### Generate Domain ###
dom = WindSE.BoxDomain(options)

### Save and Plot the domain ###
dom.Save()
dom.Plot()