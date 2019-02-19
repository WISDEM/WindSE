### This won't be needed once the package is part of local python path ###
import sys
sys.path.insert(0, '../dev/')



import WindSE

### Create an Instance of the Options ###
options = WindSE.Options()

### Generate Wind Farm ###
farm = WindSE.GridWindFarm(options)

farm.PrintLocations()


