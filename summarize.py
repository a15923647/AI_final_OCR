import os
import re
import sys
import json
import pprint
from collections import OrderedDict
prefix = sys.argv[1]
res_dict = OrderedDict()
for dir in [f for f in os.listdir() if os.path.isdir(f)]:
    for f in os.listdir(dir):
        match_res = re.match(f"^{prefix}.*eval_res.json$", f)
        full_path = os.path.join(dir, f)
        if match_res:
            with open(full_path, 'r') as fin:
                print(full_path)
                res_d = json.load(fin)
            for k, d in res_d.items():
                del d['wrong_cases']
                print(d)
            res_dict.update({full_path: res_d})
            
pprint.pprint(res_dict)
with open('summary.json', 'w') as fout:
    json.dump(res_dict, fout, indent=4)