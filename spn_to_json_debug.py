import os
from pathlib import Path
import pandas as pd
import numpy as np
from spn.io.file import from_file
from spn.utils.evidence import Evidence

dataset = 'nltcs'
spn_file = Path(f"20-datasets/{dataset}/{dataset}.spn")
spn = from_file(spn_file)

# Recreate the exact input you are giving to the spn on the neupi side
vars = spn.scope()
assignment = Evidence({
    vars[0]: [1],
    vars[1]: [0],
    vars[2]: [0],
    vars[3]: [0],
    vars[4]: [0],
    vars[5]: [0],
    vars[6]: [1],
    vars[7]: [0], 
    vars[8]: [0],
    vars[9]: [0],
    vars[10]: [0],
    vars[11]: [0],
    vars[12]: [1],
    vars[13]: [0],
    vars[14]: [1], 
    vars[15]: [0]
})

# Enter the spn here to debug and see where this one and the neupi one stray 
# from each other
spn.value(assignment)