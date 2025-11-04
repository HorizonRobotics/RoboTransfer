import os
import sys

cur_dir = os.path.abspath(__file__).split('paths')[0]

python_paths = [
    os.path.join(cur_dir, 'diffusers/src'),
]

# print("python_paths: ", python_paths)

### pip uninstall diffusers
# diffusers-0.35.1

for python_path in python_paths:
    sys.path.insert(0, python_path)
    if 'PYTHONPATH' in os.environ:
        os.environ['PYTHONPATH'] += ':{}'.format(python_path)
    else:
        os.environ['PYTHONPATH'] = python_path