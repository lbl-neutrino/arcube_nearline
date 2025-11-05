#!/usr/bin/env python

import argparse
import glob
from pathlib import Path
import time
from typing import Callable, Optional
import pandas as pd
import sqlite3
import numpy as np
import os
import re
import time
import h5py

def find_CRO_LRO_filematches(runsdb_path, saveDB_CSV=False):
    # access runs DB
    conn = sqlite3.connect(runsdb_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # access tables
    df_gsubruns = pd.read_sql_query(f"SELECT * FROM {'All_global_subruns'}", conn)
    df_crssummary = pd.read_sql_query(f"SELECT * FROM {'CRS_summary'}", conn)
    df_lrssummary = pd.read_sql_query(f"SELECT * FROM {'LRS_summary'}", conn)

    # grab info from LRS_summary table
    lrs_files = df_lrssummary.filename.to_numpy()
    lrs_run = df_lrssummary.run.to_numpy()
    lrs_subrun = df_lrssummary.subrun.to_numpy()

    # grab info from global table, mask out None values
    global_lrs_subrun = df_gsubruns.lrs_subrun.to_numpy()
    global_lrs_run = df_gsubruns.lrs_run.to_numpy()
    global_crs_subrun = df_gsubruns.crs_subrun.to_numpy()
    
    None_mask = ~np.equal(global_lrs_subrun, None) & ~np.equal(global_crs_subrun, None) & ~np.equal(global_lrs_run, None)
    global_lrs_subrun = global_lrs_subrun[None_mask].astype('int')
    global_run = global_lrs_run[None_mask].astype('int')
    global_crs_subrun = global_crs_subrun[None_mask].astype('int')

    # Find CRS run numbers for each LRO filename
    global_lrs_run_subrun = np.hstack((global_run[:, np.newaxis],
                                       global_lrs_subrun[:, np.newaxis]))
    lrs_run_subrun = np.hstack((lrs_run[:, np.newaxis],
                                lrs_subrun[:, np.newaxis]))
    
    A = global_lrs_run_subrun
    B = lrs_run_subrun
    A_view = A.view([('', A.dtype)] * A.shape[1])
    B_view = B.view([('', B.dtype)] * B.shape[1])
    
    # Collect all matches (LRO → multiple CRS matches)
    lrs_to_global_indices = [np.flatnonzero(A_view == b) for b in B_view]
    
    # Build arrays for LRO ↔ CRS matches
    lrs_file_matches = []
    crs_run_matches = []
    crs_subrun_matches = []
    
    for i, match_indices in enumerate(lrs_to_global_indices):
        if match_indices.size:
            for idx in match_indices:
                lrs_file_matches.append(lrs_files[i])
                crs_run_matches.append(global_run[idx])
                crs_subrun_matches.append(global_crs_subrun[idx])
    
    lrs_file_matches = np.array(lrs_file_matches)
    crs_run_matches = np.array(crs_run_matches)
    crs_subrun_matches = np.array(crs_subrun_matches)
    
    # Find CRS filenames corresponding to those matches
    crs_subrun = df_crssummary.subrun.to_numpy()
    crs_run = df_crssummary.global_run.to_numpy()
    crs_filename = df_crssummary.filename.to_numpy()
    
    crs_run_subrun = np.hstack((crs_run[:, np.newaxis], crs_subrun[:, np.newaxis]))
    crs_run_subrun_matches = np.hstack((crs_run_matches[:, np.newaxis],
                                        crs_subrun_matches[:, np.newaxis]))
    
    A = crs_run_subrun
    B = crs_run_subrun_matches
    A_view = A.view([('', A.dtype)] * A.shape[1])
    B_view = B.view([('', B.dtype)] * B.shape[1])
    
    # Again, collect all matches (each LRO may map to multiple CRS filenames)
    crs_indices_per_lro = [np.flatnonzero(A_view == b) for b in B_view]
    
    lrs_file_final = []
    crs_filename_matches = []
    
    for i, match_indices in enumerate(crs_indices_per_lro):
        if match_indices.size:
            for idx in match_indices:
                lrs_file_final.append(lrs_file_matches[i])
                crs_filename_matches.append(crs_filename[idx])
    
    lrs_file_final = np.array(lrs_file_final)
    crs_filename_matches = np.array(crs_filename_matches)

    if saveDB_CSV:
        DF = pd.DataFrame(np.hstack((lrs_file_final[:, np.newaxis], crs_filename_matches[:, np.newaxis])))
        DF.to_csv('DB_file_matches.csv', index=False)

    return lrs_file_final, crs_filename_matches

def SearchDirForFiles(directory, file_ending):
    root_dir = Path(directory)
    files = list(root_dir.rglob(f"*{file_ending}"))
    return files

def ExtractLRORunSubRun(filenames):
    # Extract run and subrun from LRO filenames
    # Input: List of filenames
    # Output: List of (run, subrun) combos (empty list if no match found)
    lro_pattern = re.compile(r'(\d+_p\d+)')
    lro_file_run_subrun = []
    for f in filenames:
        found_pattern = lro_pattern.search(f)
        if found_pattern is not None:
            found_pattern_split = found_pattern.group(1).split('_p')
            lro_file_run_subrun.append([int(found_pattern_split[0]), int(found_pattern_split[1])])
        else:
            lro_file_run_subrun.append([])
    return lro_file_run_subrun

def ExtractCRODate(filenames):
    # Extract date from CRO filename
    # Input: List of filenames
    # Output: List of dates (YYYY_mm_DD_HH_MM_SS format), empty str if no match found
    cro_pattern = re.compile(r'\b\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}')
    cro_file_date = []
    for f in filenames:
        found_pattern = cro_pattern.search(f)
        if found_pattern is not None:
            cro_file_date.append(found_pattern.group(0))
        else:
            cro_file_date.append('')

    return cro_file_date

def main():
    ### This script looks in the runsDB for new charge/light file combos, then looks in the charge/light directories
    ### to see if these files exist. If they do, a text file is created for every light file with the list of charge
    ### files that match to it and also exist. 
    ###
    ### Note that this script does not check whether the files are "complete"
    ### or not, that is the responsibility of whatever script uses the text files as input.
    
    ap = argparse.ArgumentParser()
    #ap.add_argument('prog', type=Path)
    ap.add_argument('--charge_dir_path', type=Path)
    ap.add_argument('--light_dir_path', type=Path)
    ap.add_argument('--runsdb_path', type=Path)
    ap.add_argument('--file_ending', help='File ending to search for in charge and light dirs (if not .hdf5)')
    ap.add_argument('--sleep_between_scans', type=int, default=30,
                    help='Minutes to sleep between filesystem/DB scans')
    
    args = ap.parse_args()
    file_ending = '.hdf5'
    if args.file_ending:
        file_ending = args.file_ending
    
    while True:
        #### find matched LRO and CRO files from runs DB
        lro_filenames, cro_filenames = find_CRO_LRO_filematches(args.runsdb_path, saveDB_CSV=False)
        lro_file_run_subrun = ExtractLRORunSubRun(lro_filenames)
        cro_file_dates = ExtractCRODate(cro_filenames)
    
        #### search charge and light directories for produced files
        found_cro_filepaths = SearchDirForFiles(args.charge_dir_path, file_ending)
        found_lro_filepaths = SearchDirForFiles(args.light_dir_path, file_ending)
        found_cro_filenames = [os.path.basename(file) for file in found_cro_filepaths]
        found_lro_filenames = [os.path.basename(file) for file in found_lro_filepaths]
    
        # find run and subrun for found LRO files
        found_lro_file_run_subrun = ExtractLRORunSubRun(found_lro_filenames)
        # find dates for found CRO files
        found_cro_file_dates = ExtractCRODate(found_cro_filenames)
    
        # look for matched charge/light file combos that have been produced already
        matched_file_indices = []
        for i in range(len(cro_file_dates)):
            lro_run_subrun = lro_file_run_subrun[i]
            cro_date = cro_file_dates[i]
            if cro_date and lro_run_subrun and cro_date in found_cro_file_dates \
                        and lro_run_subrun in found_lro_file_run_subrun:
                #print(f'CRO date = {cro_date}; LRO run/subrun = {lro_run_subrun}')
                for j in range(len(found_cro_file_dates)):
                    if found_cro_file_dates[j] == cro_date:
                        for k in range(len(found_lro_file_run_subrun)):
                            if found_lro_file_run_subrun[k] == lro_run_subrun:
                                matched_file_indices.append([j, k])
                                break
        
        matched_file_indices = np.array(matched_file_indices)
        
        found_matched_cro_filepaths = np.array(found_cro_filepaths)[matched_file_indices[:,0]]
        found_matched_lro_filepaths = np.array(found_lro_filepaths)[matched_file_indices[:,1]]
        
        # make text file for each LRO file with matched CRO files
        for i, lro_filepath in enumerate(found_matched_lro_filepaths):
            text_filename = os.path.basename(lro_filepath).split('.')[0] + '.INPUTS.txt'
            text_filepath = os.path.join(os.path.dirname(lro_filepath), text_filename)
            if os.path.exists(text_filepath):
                with open(text_filepath, "r") as f:
                    lines = [line.strip() for line in f.readlines()]
                    if str(found_matched_cro_filepaths[i]) in lines:
                        continue
                        
            with open(text_filepath, "a") as f:
                f.write(str(found_matched_cro_filepaths[i]) + '\n')

        print(f'Sleeping {args.sleep_between_scans} minutes until checking for new files in filesystem/runsDB.')
        time.sleep(args.sleep_between_scans*60)
                
if __name__ == '__main__':
    main()
