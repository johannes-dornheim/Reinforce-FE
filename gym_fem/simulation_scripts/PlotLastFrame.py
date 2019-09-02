# -------------------- File Has to be opened by the Abaqus Python Interpreter! --------------------
# SetContourRange: https://imechanica.org/node/17034
# -------------------------------------------------------------------------------------------------
# noinspection PyUnresolvedReferences
from abaqus import *
from odbAccess import openOdb, isUpgradeRequiredForOdb, upgradeOdb, OdbError
from abaqusConstants import *
import visualization
import sys
import shutil
from math import fabs, ceil, floor, copysign, log10



def SetContourRange(minval, maxval, vp='Viewport: 1', minIntervals=10):
    """
    Sets up nice contour ranges in ABAQUS/CAE
    Written by Lance C. Hibbeler, August 2014
    """

    minval = float(minval)  # ensure minval is float
    maxval = float(maxval)  # ensure minval is float

    r = maxval - minval  # range of input
    d = int(round(log10(r))) - 1  # next lowest power of 10 of range

    # find appropriate interval size from {1, 0.5, 0.25, 0.2}
    for i in [1, 2, 4, 5]:
        numIntervals = int(round(float(i) * r / (10.0 ** d)))
        if numIntervals >= minIntervals:
            interval = (10.0 ** d) / float(i)
            break

    # find first interval larger than minval
    if (minval < 0.0) and (maxval > 0):
        s = 1.0
    else:
        s = 0.0
    imin = copysign((ceil(fabs(minval) / interval) - s) * interval, minval)

    # find first interval smaller than maxval
    imax = copysign(floor(fabs(maxval) / interval) * interval, maxval)

    # define intervals
    ival = imin
    intervals = [ival]
    build = True
    while (build == True):
        ival = ival + interval
        if (ival > imax):
            build = False
        else:
            intervals.append(ival)

    # setup abaqus viewport
    co = session.viewports[vp].odbDisplay.contourOptions
    co.setValues(intervalType=USER_DEFINED, intervalValues=intervals)

    return


# argparse seems not to work here (first arguments are abaqus internals)
odb_path = sys.argv[-2]
png_path = sys.argv[-1]

# Create a viewport for this example.

myViewport = session.Viewport(name=
                              'State Plot', origin=(0, 0),
                              width=250, height=200)
SetContourRange(0, 800, vp='State Plot')

# upgrade odb if required (original simulation executed in older abaq. version)
if isUpgradeRequiredForOdb(odb_path):
    upgradeOdb(odb_path, odb_path + '_')
    shutil.move(odb_path + '_.odb', odb_path)

# Open the output database and associate it with the viewport.
# Then set the plot state to CONTOURS_ON_DEF
try:
    odb = visualization.openOdb(path=odb_path)

except (AbaqusException), e:
    print 'Abaqus Exception:', e

myViewport.setValues(displayedObject=odb)

myViewport.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF,))

# Establish print preferences
session.printOptions.setValues(vpBackground=OFF)
session.psOptions.setValues(orientation=LANDSCAPE)
myViewport.viewportAnnotationOptions.setValues(
    triad=OFF, title=OFF, state=OFF)
myViewport.odbDisplay.basicOptions.setValues(
    coordSystemDisplay=OFF, )

# Find the last available Step / Frame
ordered_stepnames = odb.steps.keys()
last_step = ordered_stepnames[-1]
numFrames = len(odb.steps[last_step].frames)

if last_step is not None and numFrames > -1:
    #   Display a contour plot.
    #   Display the step description and the increment number.

    myViewport.odbDisplay.setFrame(step=last_step, frame=numFrames - 1)
    myViewport.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF,))
    session.printToFile(png_path, PNG, (myViewport,))
else:
    print "no steps present"
