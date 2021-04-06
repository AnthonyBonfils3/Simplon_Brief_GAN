#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Mar 24 16:00:37 2021

@author: bonfils
"""

import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from app import app, server
import datetime
import pickle
from PIL import Image
import base64
from io import BytesIO
import numpy as np
from .fonctions import load_generator_model
import tensorflow as tf

batch_size = 32
latent_dim = 128


## Load Image Generation model
model_path = './api/data/generator.h5'
model_gen = load_generator_model(model_path)


## Generate image #-> Regarder https://stackoverflow.com/questions/62101497/plotly-dash-how-to-generate-html-components-with-a-for-loop
def generate_images(batch_size, latent_dim):
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    generated_images = model_gen.predict(random_latent_vectors)
    # dbc.Col(dbc.Card(children=[html.H5(#children='Chicken',
    #                                    className="text-center"),
    #                            html.Img(src="/assets/chicken.jpg", height="70px")]),),
    return generated_images

#generated_images = generate_images(batch_size, latent_dim)

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Welcome to the Avatar Generation API", className="text-center")
                    , className="mb-4 mt-4")
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='Here you can download unique Avatars or Generate any Avatars you want :')                        
                    , className="mb-4")
            ]),
        dbc.Row([generate_images(1, latent_dim) for i in range(batch_size)
            
            # dbc.Col(dbc.Card(children=[html.H5(#children='Chicken',
            #                                    className="text-center"),
            #                            html.Img(src="/assets/chicken.jpg", height="70px")]),),
            # dbc.Col(dbc.Card(children=[html.H5(#children='Monkey',
            #                                    className="text-center"),
            #                            html.Img(src="/assets/monkey.jpg", height="70px")]),),
            # dbc.Col(dbc.Card(children=[html.H5(#children='Bear',
            #                                    className="text-center"),
            #                            html.Img(src="/assets/bear.jpg", height="70px")]),),
            # dbc.Col(dbc.Card(children=[html.H5(#children='Pandas',
            #                                    className="text-center"),
            #                            html.Img(src="/assets/panda.jpg", height="70px")]),),
            # dbc.Col(dbc.Card(children=[html.H5(#children='Deer',
            #                                    className="text-center"),
            #                            html.Img(src="/assets/deer.jpg", height="70px")]),),
            # dbc.Col(dbc.Card(children=[html.H5(#children='Eagle',
            #                                    className="text-center"),
            #                            html.Img(src="/assets/eagle.jpg", height="70px")]),),
            # dbc.Col(dbc.Card(children=[html.H5(#children='Elephant',
            #                                    className="text-center"),
            #                            html.Img(src="/assets/elephant.jpg", height="70px")]),),
            # dbc.Col(dbc.Card(children=[html.H5(#children='Spider',
            #                                    className="text-center"),
            #                            html.Img(src="/assets/spider.jpg", height="70px")]),
            #                            className="mb-4"),
            ]),
        dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'solid',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'background-color':'gray'
        },
        # Allow multiple files to be uploaded
        multiple=False
        ),
        html.Div(id='output-image-upload', className="mb-4"),
        html.A("Get the full code of app on my github repositary",
               href="https://github.com/AnthonyBonfils3/")
])])
def load_and_preprocess(image):
    encoded_image = image.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)
    bytes_image = BytesIO(decoded_image)
    image = Image.open(bytes_image).convert('RGB')
    impred = image.resize((80,80))
    return impred

def np_array_normalise(test_image):
   np_image = np.array(test_image)
   final_image = np.expand_dims(np_image, 0)
   return final_image

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),)

def prediction(image):
    if image is not None:
        final_img = load_and_preprocess(image)
        final_img = np_array_normalise(final_img)
        with open('/Users/buu/devia/brief-04-02-animalClassification/hog_sgd_model.pkl', 'rb') as f1:
            pred = pickle.load(f1)
        impred = pred.predict(final_img)
        return html.Div([
            html.H1(impred) ,
            html.Img(src=image, style={'height':'30%', 'width':'30%'}),
            html.Hr(),
            ], className="text-center")


