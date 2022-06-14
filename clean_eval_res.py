import os
import sys
for dir in [f for f in os.listdir() if os.path.isdir(f)]:
    fname = "eval_res.json" if len(sys.argv) == 1 else sys.argv[1]
    if fname != '*':
        target = os.path.join(dir, fname)
        if os.path.exists(target):
            os.remove(target)
    else:
        for f in os.listdir(dir):
            if f.endswith("eval_res.json"):
                target = os.path.join(dir, f)
                os.remove(target)