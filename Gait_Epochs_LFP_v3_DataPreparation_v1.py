import mne
import os
import math
import numpy as np
import csv
import re
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.signal import spectrogram, resample
from scipy.signal import hilbert
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib as mpl
from scipy.interpolate import interp1d
import pickle
import mne_icalabel
from pactools import Comodulogram
import tensorpac
from statsmodels.formula.api import ols
import shutil

def interpolate_significant_zeros_in_raw(raw, eps=1e-6):
    data = raw.get_data()
    n_channels, n_times = data.shape
    time_idx = np.arange(n_times)
    near_zero_mask = np.abs(data) < eps  # shape: (n_channels, n_times)
    
    channels_near_zero = np.sum(near_zero_mask, axis=0)
    time_bad = np.where(channels_near_zero >= (n_channels / 3))[0]
    
    print("Time indices flagged for interpolation:", len(time_bad))
    valid_idx = np.setdiff1d(time_idx, time_bad)
    print("Using these valid indices for interpolation:", len(valid_idx))
    
    for ch in range(n_channels):
        if raw.ch_names[ch] == 'Trigger': continue
        channel_data = data[ch, :]
        data[ch, time_bad] = np.interp(time_bad, valid_idx, channel_data[valid_idx])
    
    raw._data = data
    return raw


# In[7]:
def mark_bad_gaps(raw):
    # Extract events and event IDs from annotations
    events, event_id = mne.events_from_annotations(raw)

    # Check if '<Gap Begin>' and '<Gap End>' exist
    if '<Gap Begin>' in event_id and '<Gap End>' in event_id:
        gap_begin_id = event_id['<Gap Begin>']
        gap_end_id = event_id['<Gap End>']

        # Extract onset times for gap events
        gap_begins = events[events[:, 2] == gap_begin_id][:, 0] / raw.info['sfreq']
        gap_ends = events[events[:, 2] == gap_end_id][:, 0] / raw.info['sfreq']

        # Ensure equal number of gap begins and ends
        if len(gap_begins) != len(gap_ends):
            print("Mismatched number of '<Gap Begin>' and '<Gap End>' events. Trimming excess events.")
            min_len = min(len(gap_begins), len(gap_ends))
            gap_begins = gap_begins[:min_len]
            gap_ends = gap_ends[:min_len]

        # Create annotations for each gap
        bad_gap_annotations = []
        for start, end in zip(gap_begins, gap_ends):
            duration = end - start
            if duration > 0:  # Only mark valid gaps
                bad_gap_annotations.append(dict(onset=start, duration=duration, description='BAD_gap'))

        # Add the annotations to the raw object
        gap_annotations = mne.Annotations(
            onset=[gap['onset']-0.5 for gap in bad_gap_annotations],
            duration=[gap['duration']+1.0 for gap in bad_gap_annotations],
            description=[gap['description'] for gap in bad_gap_annotations],
            orig_time=raw.annotations.orig_time,  # Match the orig_time to the existing annotations
        )
        raw.set_annotations(raw.annotations + gap_annotations)
        print("Bad gaps marked successfully.")
    else:
        print("No '<Gap Begin>' or '<Gap End>' events found in annotations.")

    return raw

def set_channel_types(edf):
    result = edf.copy()
    ch_types_map = {}
    for i in result.ch_names:
        if i.isdigit() or re.match("[0-9]*-[0-9]*",i):
            ch_types_map[i] = "seeg"
        if "ECG" in i:
            ch_types_map[i] = "ecg"
        if "EMG" in i:
            ch_types_map[i] = "emg"
        if re.match("[LR][0-9]", i):
            ch_types_map[i] = "dbs"
    print(ch_types_map)
    result.set_channel_types(ch_types_map)
    return result
def find_file_path(all_files_list, patient, filename):
    for root, dirs, files in all_files_list:
        for f in files:
            if patient in root and f == filename:
                path = os.path.join(root, f)
                return path
    return None
def process_edf_with_bipolar_reference(edf, infos):
    # Get channel types from the EDF
    ch_types = edf.get_channel_types()
    # Check if there are DBS channels
    if 'dbs' in ch_types:
        anodes = ['L0', 'L1', 'L2', 'R0', 'R1', 'R2']
        cathodes = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']
        
        # Remove bad channels from anodes and cathodes
        if infos.get('Bads') is not None and infos['Bads'] is not np.nan:
            for bad_ch in infos['Bads'].split(','):
                print(f"Remove {bad_ch} from EDF LFP channels")
                if bad_ch in anodes:
                    anodes.remove(bad_ch)
                if bad_ch in cathodes:
                    cathodes.remove(bad_ch)
        
        # Create bipolar channel names
        ch_names = [f"{a}_b" for a in anodes]
        # Set bipolar reference
        edf = mne.set_bipolar_reference(edf, anode=anodes, cathode=cathodes, ch_name=ch_names,
                                        drop_refs=False)
    # Print final channel names
    print(edf.ch_names)
    return edf

def scale_channels(raw):
    ch_to_scale = []
    for ch in raw.ch_names:
        if ch == 'Trigger': continue
        if raw.copy().pick(ch).get_data().max() > 1000.0 or raw.copy().pick(ch).get_data().min() < -1000.0: 
            ch_to_scale.append(raw.ch_names.index(ch))
    print([edf.ch_names[x] for x in ch_to_scale])
    chs = ch_to_scale
    raw_scaled = raw.copy()
    if len(chs) == 0: return raw_scaled
    scaling_factor = 10e-7
    for ch_idx in chs:
        ch_name = raw.info['chs'][ch_idx]['ch_name']
        first_value_before = raw._data[ch_idx, 0:10]
        raw_scaled._data[ch_idx] *= scaling_factor
        first_value_after = raw_scaled._data[ch_idx, 0:10]
    return raw_scaled

def update_events(raw):
    for i, desc in enumerate(raw.annotations.description):
        if desc == '24': raw.annotations.description[i] = '50'
        if desc == '30': raw.annotations.description[i] = '60'
        if desc == '100': raw.annotations.description[i] = '50'
        if desc == '120': raw.annotations.description[i] = '60'
        if desc == '108': raw.annotations.description[i] = '54'
        if desc == '104': raw.annotations.description[i] = '54'
        if desc == '96': raw.annotations.description[i] = '54'
        if desc == '128': raw.annotations.description[i] = '64'
    triggers = mne.find_events(raw, min_duration=0.01)
    events, event_id = mne.events_from_annotations(raw)
    trigger_id_maps = {100034: '60', 36864: '64', 61354: '50', 129255: '54', 
                       100096: '60', 62025: '50', 129865: '54', 82511: 'Rest',
                       82091: '28', 111311: '22', 43410: '18', 32133:'56',
                       13824: '50', 115584: '60', 47744: '26', 82267: '26',
                       14190: '50', 48140: '26', 13701: '50', 115553: '60',
                       18432: '32', 47652: '26', 72435: '60', 52065: '54',
                       38485: '50', 86016: '64', 58855: '112', 100001: 'Rest',
                       45275: '54', 31695: '54', 10294: 'Sit', 38451: '50', 31670:'54',
                       45232: '54', 114240: '106', 99958: '106', 45312: '54', 38460: '50' }
    id_maps = {}
    for key,value in trigger_id_maps.items():
        if value in event_id.keys():
            id_maps[key] = event_id[value]
    target_ids = ['50', '54', '60', '64']
    remove_ids = []
    for i, t in enumerate(triggers):
        try:
            t[2] = id_maps[t[2]]
        except:
            import traceback
            traceback.print_exc()
            print(mne.events_from_annotations(raw)[1])
            for i, j in zip(mne.find_events(raw), mne.events_from_annotations(raw)[0]):
                print(i,j)
            exit()
    return triggers, event_id
    
def find_sfp_file(locator_dir, patient):
    # Loop through all .sfp files in the directory
    for file_name in os.listdir(locator_dir):
        # Check if the file is a .sfp file
        if file_name.endswith('.sfp'):
            # Check if the patient name is a substring of the file name
            if patient.lower() in file_name.lower():
                sfp_fpath = os.path.join(locator_dir, file_name)
                print(f"Found matching .sfp file: {sfp_fpath}")
                return sfp_fpath

    # If no matching file is found, return None
    print(f"No matching .sfp file found for patient: {patient}")
    return None

name_maps = {
    '1': 'Fp1', '2': 'Fpz', '3': 'Fp2', '4': 'AF7', '5': 'AF3', '6': 'AF4', '7': 'AF8', '8': 'F7', 
    '9': 'F5', '10': 'F3', '11': 'F1', '12': 'Fz', '13': 'F2', '14': 'F4', '15': 'F6', '16': 'F8', 
    '17': 'FC5', '18': 'FC3', '19': 'FC1', '20': 'FCz', '21': 'FC2', '22': 'FC4', '23': 'FC6', '24': 'C5',
    '25': 'C3', '26': 'C1', '27': 'Cz', '28': 'C2', '29': 'C4', '30': 'C6', '31': 'CP5', '32': 'CP3',
    '1-2': 'CP1', '2-2': 'CPz', '3-2': 'CP2', '4-2': 'CP4', '5-2': 'CP6', '6-2': 'P7', '7-2': 'P5',
    '8-2': 'P3', '9-2': 'P1', '10-2': 'Pz', '11-2': 'P2', '12-2': 'P4', '13-2': 'P6', '14-2': 'P8',
    '15-2': 'PO7', '16-2': 'PO5', '17-2': 'PO3', '18-2': 'POz', '19-2': 'PO4', '20-2': 'PO6', '21-2': 'PO8',
    '22-2': 'O1', '23-2': 'Oz', '24-2': 'O2', '25-2': 'L0', '26-2': 'L1', '27-2': 'L2', '28-2': 'L3',
    '29-2': 'R0', '30-2': 'R1', '31-2': 'R2', '32-2': 'R3', '33': 'ECG', '34': 'LEMG', '36': 'REMG',
    '37': 'LEMG-ARM', '38': 'REMG-ARM', '39': 'LEMG-AT', '40': 'REMG-AT', 'Trigger': 'Trigger'
}
root_path = '/Volumes/Workspace/NextCloud/Ruijin/PD EEG Analysis'
workspace = '/Volumes/Workspace/Data/PD EEG Analysis/Dataspace'
index_PD_Gait = pd.read_excel(os.path.join(root_path, 'Record_v2.xlsx'), index_col=1)
# index_PD_Gait = pd.read_excel(os.path.join('Record_DBS.xlsx'), index_col=1)
print(index_PD_Gait)
locator_dir   = '/Volumes/Workspace/Data/LOCATOR'

for patient, infos in index_PD_Gait.iterrows():
    if not infos['Useable']: continue
    if infos['Preprocessed']: continue
    print('='*80)
    subject_folder = os.path.join(workspace, patient)
    if not os.path.exists(subject_folder): os.mkdir(subject_folder)
    raw_save_path  = os.path.join(workspace, patient, f"{infos['Type']}-raw.fif")
    file_path = find_file_path(os.walk(root_path), patient.replace('_LFP',''), infos['EEG/LFP'])
    print("-"*80)
    print(f"{patient} EDF: {file_path}")
    
    edf = mne.io.read_raw_edf(file_path, preload=True)
    edf = interpolate_significant_zeros_in_raw(edf)
    edf_save_path = os.path.join(workspace, patient, f"{infos['Type']}-orig.edf")
    print(f'Save EDF to file {edf_save_path}.')
    shutil.copyfile(file_path, edf_save_path)
    edf = edf.crop(edf.times[0], edf.times[-1]-10.0)
    if '1-2' in edf.ch_names: edf.rename_channels(name_maps)
    edf = set_channel_types(edf)
    if 'A1' in edf.ch_names: edf.drop_channels(['A1', 'A2'])
    if 'O1_Broken' in edf.ch_names: edf.drop_channels(['O1_Broken'])
    edf = mark_bad_gaps(edf)
    # anno, bads = mne.preprocessing.annotate_amplitude(edf, peak={'eeg': 400e-6}, picks='eeg', verbose=True)
    # edf.set_annotations(edf.annotations + anno)
    anno = edf.annotations
    bad_annotations = [annot for annot in anno if "BAD" in annot['description']]
    # print(f"Peak rejects: {len([x for x in bad_annotations if x['description'] == 'BAD_peak'])}")
    # print(f"Peak rejects seconds: {np.sum([x['duration'] for x in bad_annotations if x['description'] == 'BAD_peak'])}")
    print(f"Gap rejects: {len([x for x in bad_annotations if x['description'] == 'BAD_gap'])}")
    print(f"Gap rejects seconds: {np.sum([x['duration'] for x in bad_annotations if x['description'] == 'BAD_gap'])}")
    print("-"*80)
    edf = process_edf_with_bipolar_reference(edf, infos)
    edf = scale_channels(edf)
    edf = edf.notch_filter([50.0, 100.0, 150.0, 200.0, 250.0], notch_widths=2)
    edf = edf.filter(1.0, 250.0)
    edf, _ = mne.set_eeg_reference(edf, ref_channels='average', projection=True)
    events, event_id = update_events(edf)
    print(events.shape)
    if edf.info['sfreq'] != 512.0: edf, events = edf.resample(512.0, events=events)
    # montage = mne.channels.make_standard_montage('standard_1020')
    print("-"*80)
    if '_LFP' in patient: continue
    # sfp_fpath = os.path.join(locator_dir, patient + '.sfp')
    sfp_fpath = find_sfp_file(locator_dir, patient.split('_')[1])
    montage   = mne.channels.read_custom_montage(sfp_fpath)
    edf.set_montage(montage, on_missing='ignore', match_case=False)
    montage_info = [f"{x['ch_name']} {x['loc'][:3]}" for x in edf.info['chs']]
    for x in montage_info: print(x)
    
    # 6. Load and interpolate bad channels
    if infos['Bads'] is not np.nan:
        print(f"Interpolate bad channels: [{infos['Bads']}]", end="")
        edf.info['bads'] = [x for x in infos['Bads'].split(',')]
        edf = edf.interpolate_bads()
    
    # 7. ICA EMG Removal
    print("Construct and fitting ICA......", end="")
    n_components = len(edf.copy().pick_types(eeg=True).ch_names)-2
    ica = mne.preprocessing.ICA(n_components=n_components, method="infomax", 
                                max_iter="auto", random_state=97, 
                                fit_params=dict(extended=True))
    ica.fit(edf, picks='eeg')
    ic_labels = mne_icalabel.label_components(edf, ica, method="iclabel")
    labels = ic_labels["labels"]
    standard_excludes = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    brain_excludes    = [idx for idx, label in enumerate(labels) if label not in ["brain"]]
    muscle_excludes   = ica.find_bads_muscle(edf)[0]
    emg_excludes      = ica.find_bads_eog(edf, 'LEMG')[0] + ica.find_bads_eog(edf, 'REMG')[0] \
                        + ica.find_bads_eog(edf, 'LEMG-AT')[0] + ica.find_bads_eog(edf, 'REMG-AT')[0]
    emg_excludes      = list(set(emg_excludes))
    strict_exludes    = list(set(standard_excludes + emg_excludes))
    print(f"Standard Excludes: {standard_excludes}")
    print(f"Brain Excludes: {brain_excludes}")
    print(f"Muscles Excludes: {muscle_excludes}")
    print(f"EMG Excludes: {emg_excludes}")
    print(f"Strict Excludes: {strict_exludes}")
    brain_edf  = edf.copy()
    strict_edf = edf.copy()
    ica.apply(edf, exclude=standard_excludes)
    print("Success!", end="")
    
    strict_edf = ica.apply(strict_edf, exclude=strict_exludes)
    strict_edf.save(os.path.join(workspace, patient, f"{infos['Type']}-strict-raw.fif"), overwrite=True)
    strict_epochs = mne.Epochs(strict_edf, events=events, event_id=event_id, event_repeated='drop',
                        on_missing='ignore', tmin=-1.5, tmax=3.0,)
    strict_epochs.load_data()
    strict_epochs.save(os.path.join(workspace, patient, f"{infos['Type']}-strict-epo.fif"), overwrite=True)
    
    brain_edf = ica.apply(brain_edf, exclude=brain_excludes)
    brain_edf.save(os.path.join(workspace, patient, f"{infos['Type']}-brain-raw.fif"), overwrite=True)
    brain_epochs = mne.Epochs(brain_edf, events=events, event_id=event_id, event_repeated='drop',
                        on_missing='ignore', tmin=-1.5, tmax=3.0,)
    brain_epochs.load_data()
    brain_epochs.save(os.path.join(workspace, patient, f"{infos['Type']}-brain-epo.fif"), overwrite=True)
    # 9. Export EDF and Epochs to file
    edf.save(raw_save_path, overwrite=True)
    epochs = mne.Epochs(edf, events=events, event_id=event_id, event_repeated='drop',
                        on_missing='ignore', tmin=-1.5, tmax=3.0,)
    epochs.load_data()
    epochs.save(os.path.join(workspace, patient, f"{infos['Type']}-epo.fif"), overwrite=True)

    print(f"Finished {patient}")
print("Preprocessing finished.")

