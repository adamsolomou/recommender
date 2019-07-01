#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:50:33 2019

@author: yannis
"""

#%%
import pandas as pd
import re

def convert_to_csv(file, regex, columns, output, dry=False):

    with open(file) as f:
        text = f.read()
        results = re.findall(regex,text)
        df = pd.DataFrame(results, columns=columns)
        if dry:
            print(df)
        else:
            df.to_csv(output, index=False)

if __name__ == "__main__": 
    '''
    file = 'layers.out'
    layers_regex = 'layer: (\d+)\n(\d+\.\d+)'
    column_names = ['Layer_size', 'score']
    convert_to_csv(file, layers_regex, column_names, 'autoencoder_exp_layer.csv', dry=False)
    '''

    '''
    file = 'autoencoder_exp_depth.out'
    layers_regex = 'layers: (\[.*\]) score: (\d+\.\d+)'
    column_names = ['Architecture', 'Score']
    convert_to_csv(file, layers_regex, column_names, 'autoencoder_exp_depth.csv', dry=False)
    '''
    file = 'autoencoder_exp_masking_15_15_15.out'
    layers_regex = 'masking: (\d+\.\d+) score: (\d+\.\d+)'
    column_names = ['masking', 'Score']
    convert_to_csv(file, layers_regex, column_names, 'autoencoder_exp_masking.csv', dry=False)