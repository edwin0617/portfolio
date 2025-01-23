#!/usr/bin/env python

# Do *not* edit this script.
# These are helper functions that you can use with your code.

import os, numpy as np, scipy as sp, scipy.io, scipy.io.wavfile
import pandas as pd
from typing import Dict, Tuple, List

# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Compare normalized strings.
def compare_strings(x, y):
    try:
        return str(x).strip().casefold()==str(y).strip().casefold()
    except AttributeError: # For Python 2.x compatibility
        return str(x).strip().lower()==str(y).strip().lower()


# Find patient data files.
def find_patient_files(data_folder):
    # Find patient files.
    filenames = list()
    for f in sorted(os.listdir(data_folder)):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension=='.txt':
            filename = os.path.join(data_folder, f)
            filenames.append(filename)

    # To help with debugging, sort numerically if the filenames are integers.
    roots = [os.path.split(filename)[1][:-4] for filename in filenames]
    if all(is_integer(root) for root in roots):
        filenames = sorted(filenames, key=lambda filename: int(os.path.split(filename)[1][:-4]))

    return filenames

# Load patient data as a string.
def load_patient_data(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data

# Load a WAV file.
def load_wav_file(filename):
    frequency, recording = sp.io.wavfile.read(filename)
    return recording, frequency

# Load recordings.
def load_recordings(data_folder, data, get_frequencies=False):
    num_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:num_locations+1]

    recordings = list()
    frequencies = list()
    for i in range(num_locations):
        entries = recording_information[i].split(' ')
        recording_file = entries[2]
        filename = os.path.join(data_folder, recording_file)
        recording, frequency = load_wav_file(filename)
        recordings.append(recording)
        frequencies.append(frequency)

    if get_frequencies:
        return recordings, frequencies
    else:
        return recordings

# Get patient ID from patient data.
def get_patient_id(data):
    patient_id = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                patient_id = l.split(' ')[0]
            except:
                pass
        else:
            break
    return patient_id

# Get number of recording locations from patient data.
def get_num_locations(data):
    num_locations = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                num_locations = int(l.split(' ')[1])
            except:
                pass
        else:
            break
    return num_locations

# Get frequency from patient data.
def get_frequency(data):
    frequency = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                frequency = float(l.split(' ')[2])
            except:
                pass
        else:
            break
    return frequency

# Get recording locations from patient data.
def get_locations(data):
    num_locations = get_num_locations(data)
    locations = list()
    for i, l in enumerate(data.split('\n')):
        entries = l.split(' ')
        if i==0:
            pass
        elif 1<=i<=num_locations:
            locations.append(entries[0])
        else:
            break
    return locations

# Get age from patient data.
def get_age(data):
    age = None
    for l in data.split('\n'):
        if l.startswith('#Age:'):
            try:
                age = l.split(': ')[1].strip()
            except:
                pass
    return age

# Get sex from patient data.
def get_sex(data):
    sex = None
    for l in data.split('\n'):
        if l.startswith('#Sex:'):
            try:
                sex = l.split(': ')[1].strip()
            except:
                pass
    return sex

# Get height from patient data.
def get_height(data):
    height = None
    for l in data.split('\n'):
        if l.startswith('#Height:'):
            try:
                height = float(l.split(': ')[1].strip())
            except:
                pass
    return height

# Get weight from patient data.
def get_weight(data):
    weight = None
    for l in data.split('\n'):
        if l.startswith('#Weight:'):
            try:
                weight = float(l.split(': ')[1].strip())
            except:
                pass
    return weight

# Get pregnancy status from patient data.
def get_pregnancy_status(data):
    is_pregnant = None
    for l in data.split('\n'):
        if l.startswith('#Pregnancy status:'):
            try:
                is_pregnant = bool(sanitize_binary_value(l.split(': ')[1].strip()))
            except:
                pass
    return is_pregnant

# Get murmur from patient data.
def get_murmur(data):
    murmur = None
    for l in data.split('\n'):
        if l.startswith('#Murmur:'):
            try:
                murmur = l.split(': ')[1]
            except:
                pass
    if murmur is None:
        raise ValueError('No murmur available. Is your code trying to load labels from the hidden data?')
    return murmur


# Get pregnancy status from patient data.
def get_murmur_timing(data):
    timing = None
    for l in data.split('\n'):
        if l.startswith('#Systolic murmur timing:'):
            try:
                timing = l.split(': ')[1].strip()
            except:
                pass
    return timing


def get_murmur_locations(data):
    locations = set()
    for l in data.split('\n'):
        if l.startswith('#Murmur locations:'):
            try:
                locations = l.split(': ')[1].strip().split('+')
            except:
                pass
    if locations == ['nan']:
        locations = []
    return set(locations)


# Get outcome from patient data.
def get_outcome(data):
    outcome = None
    for l in data.split('\n'):
        if l.startswith('#Outcome:'):
            try:
                outcome = l.split(': ')[1]
            except:
                pass
    if outcome is None:
        raise ValueError('No outcome available. Is your code trying to load labels from the hidden data?')
    return outcome

# Sanitize binary values from Challenge outputs.
def sanitize_binary_value(x):
    x = str(x).replace('"', '').replace("'", "").strip() # Remove any quotes or invisible characters.
    if (is_finite_number(x) and float(x)==1) or (x in ('True', 'true', 'T', 't')):
        return 1
    else:
        return 0
    

# Santize scalar values from Challenge outputs.
def sanitize_scalar_value(x):
    x = str(x).replace('"', '').replace("'", "").strip() # Remove any quotes or invisible characters.
    if is_finite_number(x) or (is_number(x) and np.isinf(float(x))):
        return float(x)
    else:
        return 0.0


# Save Challenge outputs.
def save_challenge_outputs(filename, patient_id, classes, labels, probabilities):
    # Format Challenge outputs.
    patient_string = '#{}'.format(patient_id)
    class_string = ','.join(str(c) for c in classes)
    label_string = ','.join(str(l) for l in labels)
    probabilities_string = ','.join(str(p) for p in probabilities)
    output_string = patient_string + '\n' + class_string + '\n' + label_string + '\n' + probabilities_string + '\n'


    # Write the Challenge outputs.
    with open(filename, 'w') as f:
        f.write(output_string)


# Load Challenge outputs.
def load_challenge_outputs(filename):
    with open(filename, 'r') as f:
        for i, l in enumerate(f):
            if i==0:
                patient_id = l.replace('#', '').strip()
            elif i==1:
                classes = tuple(entry.strip() for entry in l.split(','))
            elif i==2:
                labels = tuple(sanitize_binary_value(entry) for entry in l.split(','))
            elif i==3:
                probabilities = tuple(sanitize_scalar_value(entry) for entry in l.split(','))
            else:
                break
    return patient_id, classes, labels, probabilities


# Added by th23------------------------------------------------------------------------------------------------------------------------------------------ 

def filter_wavfiles(filename):
    if "normal" in filename or "murmur" in filename or "unlabelled" in filename:
        return True
    else:
        return False 
    
def define_category(filename):
    if "normal" in filename:
        return "Absent"
    elif "murmur" in filename:
        return "Present"
    else:
        return "Unlabeled"
    
class ExponentialWarmup(object):  # Adam, 0.001, warmup_steps, default
    def __init__(self, optimizer, max_lr, rampup_length, exponent):
        self.optimizer = optimizer
        self.rampup_length = rampup_length
        self.max_lr = max_lr
        self.step_num = 1
        self.exponent = exponent

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _get_lr(self):
        return self.max_lr * self._get_scaling_factor()

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        self._set_lr(lr)

    def _get_scaling_factor(self):
        if self.rampup_length == 0:
            return 1.0
        else:
            
            current = np.clip(self.step_num, 0.0, self.rampup_length)
            phase = 1.0 - current / self.rampup_length
            return float(np.exp(self.exponent * phase * phase))    
    

def update_ema(net, ema_net, step, ema_factor):
    # update EMA model
    alpha = min(1 - 1 / step, ema_factor)
    #alpha = min(1 / step, ema_factor)
    for ema_params, params in zip(ema_net.parameters(), net.parameters()):
        ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)
    return ema_net
