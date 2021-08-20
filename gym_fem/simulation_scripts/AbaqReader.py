# -------------------- File Has to be opened by the Abaqus Python Interpreter! --------------------
# noinspection PyUnresolvedReferences
from odbAccess import openOdb, isUpgradeRequiredForOdb, upgradeOdb, OdbError
import warnings
import os
import shutil
import argparse

# todo atm. the script is simulation-specific (for 2d fem-simulation), generalize!
parser = argparse.ArgumentParser()

parser.add_argument('odb_path', help="Path of the simulation output database")
parser.add_argument('out_path', help="Path for the output CSV files")
parser.add_argument('--first_odb_path', help="Path of the first time-step output database (optional)",
                    dest='first_odb', default=None)

args = parser.parse_args()
ODB_PATH = args.odb_path
FST_ODB_PATH = args.first_odb
OUT_DIR = args.out_path
SCRIPT_VERSION = 3  # used to prevent reuse of outdated files, by extending the filename of output CSV-files

def safeOpenOdb(odb_path):
    # upgrade odb if required (original simulation executed in older abaq. version)
    if isUpgradeRequiredForOdb(odb_path):
        upgradeOdb(odb_path, odb_path+'_')
        shutil.move(odb_path+'_.odb', odb_path)
    try:
        odb = openOdb(odb_path)
    except OdbError, e:
        print str(e)
        exit(1)

    return odb


odb = safeOpenOdb(ODB_PATH)

# find first time-frame
if FST_ODB_PATH:
    first_odb = safeOpenOdb(FST_ODB_PATH)
    ordered_stepnames = first_odb.steps.keys()
    first_frame = first_odb.steps[ordered_stepnames[0]].frames[0]
else:
    ordered_stepnames = odb.steps.keys()
    first_frame = odb.steps[ordered_stepnames[0]].frames[0]

# find last frame available
ordered_stepnames = odb.steps.keys()
last_frame = odb.steps[ordered_stepnames[-1]].frames[-1]
if last_frame is None:
    warnings.warn("No frame found in " + ordered_stepnames[-1])
    exit(0)

# read out element-wise values
with open(os.path.join(OUT_DIR, 'element_extract_' + str(SCRIPT_VERSION) + '.csv'), "wt") as outFile:
    outFile.write('ID, MISES\n')
    knownlabels = []
    for elem in last_frame.fieldOutputs['S'].values:
        outFile.write('{}, {}\n'.format(str(elem.elementLabel), str(elem.mises)))
        knownlabels.append(elem.elementLabel)

# read out node-wise values
INSTANCE_DICT = {'BLECH-1': 'BLECH', 'NIEDERHALTER-1': 'NIEDERHALTER', 'MATRIZE-1': 'MATRIZE', 'STEMPEL-1': 'STEMPEL'}

with open(os.path.join(OUT_DIR, 'node_extract_' + str(SCRIPT_VERSION) + '.csv'), "wt") as outFile:
    outFile.write(
        'NODE_ID, INSTANCE, X_COORD_INITIAL, X_COORD, X_OFFSET, Y_COORD_INITIAL, Y_COORD, Y_OFFSET, TOTAL_FORCE_2\n')

    for first_frame_node, last_frame_node, last_frame_force in zip(first_frame.fieldOutputs['COORD'].values,
                                                                   last_frame.fieldOutputs['COORD'].values,
                                                                   last_frame.fieldOutputs['TF'].values):
        inst_name = first_frame_node.instance.name
        instance = INSTANCE_DICT[inst_name] if inst_name in INSTANCE_DICT.keys() else inst_name

        out_values = [first_frame_node.nodeLabel,
                      instance,
                      first_frame_node.data[0],
                      last_frame_node.data[0],
                      first_frame_node.data[0] - last_frame_node.data[0],
                      first_frame_node.data[1],
                      last_frame_node.data[1],
                      first_frame_node.data[1] - last_frame_node.data[1],
                      abs(last_frame_force.data[1])]
        outFile.write(', '.join([str(v) for v in out_values]) + '\n')
exit(0)
