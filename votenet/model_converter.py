#RUN THIS CODE AS
#cd D:\freecaed\FreeCAD 0.18\bin
#d:
#python C:\Users\adam2\Desktop\votenet-docker\votenet\model_converter.py


import FreeCAD
import Part
import Mesh

import sys

filename = sys.argv[1]

print('converting file: %s' % filename)

out_file = filename[:filename.rfind('.')] + '.stl'

shape = Part.Shape()
shape.read(filename)
doc = App.newDocument('Doc')
pf = doc.addObject("Part::Feature","MyShape")
pf.Shape = shape
Mesh.export([pf], out_file)