import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from app import app, server
import os

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1(children='Avatar Generator'), className="mb-2")
        ]),
                dbc.Row([
            dbc.Col(html.H5(children='Here you can research for animal with the keyword below')                        
                    , className="mb-4")
            ]),
        dbc.Row([
            dbc.Col(dbc.Card(children=[html.H5(children='Chicken',
                                               className="text-center"),
                                       html.Img(src="/assets/chicken.jpg", height="70px")]),),
            dbc.Col(dbc.Card(children=[html.H5(children='Monkey',
                                               className="text-center"),
                                       html.Img(src="/assets/monkey.jpg", height="70px")]),),
            dbc.Col(dbc.Card(children=[html.H5(children='Bear',
                                               className="text-center"),
                                       html.Img(src="/assets/bear.jpg", height="70px")]),),
            dbc.Col(dbc.Card(children=[html.H5(children='Panda',
                                               className="text-center"),
                                       html.Img(src="/assets/panda.jpg", height="70px")]),),
            dbc.Col(dbc.Card(children=[html.H5(children='Deer',
                                               className="text-center"),
                                       html.Img(src="/assets/deer.jpg", height="70px")]),),
            dbc.Col(dbc.Card(children=[html.H5(children='Eagle',
                                               className="text-center"),
                                       html.Img(src="/assets/eagle.jpg", height="70px")]),),
            dbc.Col(dbc.Card(children=[html.H5(children='Elephant',
                                               className="text-center"),
                                       html.Img(src="/assets/elephant.jpg", height="70px")]),),
            dbc.Col(dbc.Card(children=[html.H5(children='Spider',
                                               className="text-center"),
                                       html.Img(src="/assets/spider.jpg", height="70px")]),className="mb-4"),
            ]),
        dbc.Row([
            dbc.Col(dcc.Input(id="input1", type="text", placeholder="Type your text", debounce=True), className="mb-4 text-center")
        ]),
        dbc.Row([
            dbc.Col(html.H2(id="output"), className="mb-4 text-center")
        ]),
        html.A("Get the full code of app on my github repositary",
               href="https://github.com/Ludo-R/")
        ], className="mb-5"), 
    ])

def generate_thumbnail(image):
    return html.Div([
            html.Img(
                src = image,
                style = {
                    'height': '180px',
                    'width': '180px',
                    'float': 'left',
                    'position': 'relative',
                    'padding-top': 15,
                    'padding-right': 15,

                }
            )
        ],className="text-center")

@app.callback(
    Output("output", "children"),
    Input("input1", "value"),)
def update_output(input1):  
    
    if input1 != None:
        text = input1.capitalize()
        for subdir in os.listdir(fr'/Users/buu/devia/brief-04-02-animalClassification/Image'):
            if text == subdir[:-4]:
                print(subdir)
                current_path = os.path.join(fr'/Users/buu/devia/brief-04-02-animalClassification/Image', subdir)

                images_div = []
                for file in os.listdir(current_path):
                        image = Image.open(current_path+"/"+file).convert('RGB')
                        images_div.append(generate_thumbnail(image))
                return html.Div(images_div)
        else:
            return ""