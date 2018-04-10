# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs several instances of the model for hyperparameter tuning.
"""

import os
import subprocess

RELATIVE_MODEL_LOCATION = '../mnist_with_summaries/mnist_with_summaries.py'

# Various learning rates to try.
learning_rates = [1e-2, 1e-3, 1e-4]

# The probability at which we keep during dropout.
dropouts = [0.6, 0.8, 1.0]

for learning_rate in learning_rates:
  for dropout in dropouts:
    run_name = 'run_{:.2e}_{:.2e}'.format(learning_rate, dropout)
    logdir = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                          'tensorflow/mnist/logs/',
                          run_name)
    process = subprocess.Popen(
        [
            'python',
            RELATIVE_MODEL_LOCATION,
            '--learning_rate',
            str(learning_rate),
            '--dropout',
            str(dropout),
            '--log_dir',
            logdir
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    out, err = process.communicate('')
    retcode = process.wait()
    if retcode != 0:
      print('Error: %r' % err.decode('utf-8'))
    else:
      print('Output: %r' % out.decode('utf-8'))
