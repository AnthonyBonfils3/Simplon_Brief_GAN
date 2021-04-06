#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Mar 24 16:00:37 2021

@author: bonfils
"""

import dash
import dash_bootstrap_components as dbc

# bootstrap theme
# https://bootswatch.com/lux/
external_stylesheets = [dbc.themes.SKETCHY]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Avatar generator' 

server = app.server
app.config.suppress_callback_exceptions = True