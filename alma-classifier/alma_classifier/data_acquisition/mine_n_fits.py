"""Download .fits files from the ALMA Archive based on keywords. Uses a modified version of
   ALminer's own function download_data in order to pick out 500 random images from the 
   query. There is a cleanup script added at the end in order to delete "extra" .pickle files
   coming with the .fits files due to Python as well as larger files than 30 MB."""

import alminer
import pandas as pd
from alminer_mod import download_data_mod
import os, os.path

#'"disks around low-mass stars"' is not working and has been published as an issue at the code provider.

DIR = '<PATH TO DATA DIRECTORY>'  # Will be set to '/data' as default
KEYWORDS = ['"intermediate-mass star formation"', '"outflows, jets, feedback"', '"outflows, jets and ionized winds"', '"inter-stellar medium (ism)/molecular clouds"']

def download_routine(datadir, keywords, n_files, dryrun):
    obs_holder = pd.DataFrame()
    for i in range(len(keywords)):
        # Query and filtering
        print(" Querying with keyword : " + keywords[i])
        my_query = alminer.keysearch({'science_keyword':[keywords[i]]}, print_targets=False, tap_service='NAOJ')
        selected = my_query[my_query.obs_release_date > '2016']
        selected = selected[selected.ang_res_arcsec < 0.4]
        selected = selected.drop_duplicates(subset='obs_id').reset_index(drop=True)
        obs_holder = pd.concat([obs_holder, selected])


    print("Proceeding to download {files} fits files from the following dataframe.".format(files=n_files))
    print(obs_holder)
    print(alminer.summary(obs_holder, print_targets=False))
    download_data_mod(obs_holder, fitsonly=True, dryrun=dryrun, location=datadir, filename_must_include=[".pbcor", "_sci", ".cont"], archive_mirror='NAOJ', n_fits=n_files)

    return


def cleanup(dir):
    for root, _, files in os.walk(dir):
        for f in files:
            fullpath = os.path.join(root, f)
            if f.endswith(('.pickle')):
                os.remove(fullpath)
            elif os.path.getsize(fullpath) > 30 * 1024 * 1000:
                os.remove(fullpath)

download_routine(DIR, KEYWORDS, n_files=500, dryrun=True)
cleanup(DIR)