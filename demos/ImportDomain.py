### This won't be needed once the package is part of local python path ###
import sys
sys.path.insert(0, '../dev/')



import WindSE

### Create an Instance of the Options ###
options = WindSE.Options()

### Edit desired options ###
options.domain.type = "imported"
options.domain.path = "../meshes/gaussianhill/xml/mesh.xml.gz"

### Generate Domain ###
dom = WindSE.ImportedDomain(options)

### Save and Plot the domain ###
dom.Save(filename="a_witty_name",filetype="xml.gz")
dom.Plot()