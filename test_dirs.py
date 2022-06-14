import evaluate
import os
import re
import sys
import pprint
import json
import numpy as np
from config import test_dir_config
from config import evaluate_config

update_configs = []
def construct_update_configs(dirs):
    global update_configs
    for dir in dirs:
        if os.path.exists(dir):
            paths = [os.path.join(dir, f) for f in os.listdir(dir) if re.match('^crnn_[0-9]{6}.*pt$', f)]
            for p in paths:
                batch_has_run = int(re.match('^.*crnn_([0-9]{6}).*pt$', p).group(1))
                if batch_has_run in range(32001):
                    update_configs.append({'reload_checkpoint': p})

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main(dirs=test_dir_config['lookup_dirs']):
    construct_update_configs(dirs)
    for ud in update_configs:
        evaluate_config.update(ud)
        eval_res = evaluate.main(evaluate_config, reuse_dataset=True)
        try:
            pprint.pprint(ud['reload_checkpoint'], json.dumps(eval_res, indent=4, cls=NpEncoder))
        except:
            pass
        dirname = os.path.dirname(ud['reload_checkpoint'])
        re_res = re.search('.*/crnn_([0-9]{6}).*pt$', ud['reload_checkpoint'])
        batch_has_run = 0 if not re_res else re_res.group(1)
        output_path = os.path.join(dirname, test_dir_config['eval_res_name'])
        
        old_res = {}
        if os.path.exists(output_path):
            with open(output_path, 'r') as fin:
                old_res = json.load(fin)
        old_res.update({int(batch_has_run): eval_res})
        with open(output_path, 'w') as fout:
            json.dump(old_res, fout, cls=NpEncoder, indent=4)

if __name__ == '__main__':
    main()
