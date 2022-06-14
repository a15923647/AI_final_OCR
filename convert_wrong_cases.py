import re
import os
import json
import sys
import pprint
from dataset import labels2word
from collections import OrderedDict
eval_res_file = sys.argv[1]
annotation_path = sys.argv[2]
ann_dir = os.path.dirname(annotation_path)
real2path = dict()
with open(annotation_path, 'r') as fin:
    for line in fin:
        path, idx = line.split()
        real = re.match(f"^.*_(.+)_{idx}.jpg$", path).group(1).lower()
        full_path = os.path.join(ann_dir, path)
        real2path[real] = full_path

with open(eval_res_file, 'r') as fin:
    data = json.load(fin)
res = OrderedDict()
for batch_has_run_s, d in data.items():
    wrong_cases = d['wrong_cases']
    converted = list()
    for case in wrong_cases:
        real, pred, word_cer = case
        real = labels2word(real)
        pred = labels2word(pred)
        converted.append((real, pred, word_cer, real2path[real.lower()]))
    res[int(batch_has_run_s)] = converted
pprint.pprint(res)
dump_file_suffix = eval_res_file.replace('/', '_').replace('\\', '_')
with open(f"converted_wrong_cases_{dump_file_suffix}.json", 'w') as fout:
    json.dump(res, fout, indent=4)
