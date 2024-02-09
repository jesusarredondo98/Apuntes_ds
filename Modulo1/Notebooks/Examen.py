#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 07:10:21 2022

@author: jesusarredondo
"""
import pandas as pd
import os
import json

with open('/Users/jesusarredondo/Documents/Diplomado/Modulo1/Data/archive/column_remapping.json')\
    as json_file: 
    data_json = json.load(json_file)

company = data_json['company']
taxi_id = data_json['taxi_id']
dropoff_census_tract = data_json['dropoff_census_tract']
pickup_census_tract = data_json['pickup_census_tract']
pickup_latitude = data_json['pickup_latitude']
pickup_longitude = data_json['pickup_longitude']
dropoff_latitude = data_json['dropoff_latitude']
dropoff_longitude = data_json['dropoff_longitude']

"""Importamos los datos, previamente agrupados"""

data_path = "/Users/jesusarredondo/Documents/Diplomado/Modulo1/Data/archive/data_concat.csv"


df_entera = pd.read_csv(data_path,encoding="utf-8")

print(df_entera.head(5))
