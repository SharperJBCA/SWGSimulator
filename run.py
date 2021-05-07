import sys
import numpy as np
from matplotlib import pyplot
from SWGSimulator.Tools import Parser
from SWGSimulator.SkyModel import ska_fgchallenge
from SWGSimulator.Beams import Beams
from SWGSimulator.SimObs import SimObs


if __name__ == "__main__":
    
    parser = Parser.Parser(sys.argv[1])

    if parser['SkyModel']['do_skymodel']:
        ska_fgchallenge.main(parser)
        
    if parser['Beams']['do_beams']:
        Beams.main(parser)

    if parser['SimObs']['do_simobs']:
        SimObs.main(parser)
