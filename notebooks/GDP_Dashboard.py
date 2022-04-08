#!/usr/bin/env python

# import dependencies
import dash
import dash_bootstrap_components as dbc
import flask
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.tools as tls
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
from flask import Flask, Response



# set basic directories and filepaths for dashboard
# PROJECT_DIR = os.getcwd()
PROJECT_DIR = "C:\\Users\\Msi\\Desktop\\virtualenv_project"
DATA_DIR = os.path.join(PROJECT_DIR, 'static')

# supported locations that can be chosen within the app
AVAILABLE_AREAS = ['Campus', 'Street', 'Security', 'Alley']

# supported result plot types to show on app
PLOT_RESULT_TYPES = ['Heatmap', 'Poses']


# start flask server and dash application
server = Flask(__name__)
app = dash.Dash(__name__, server=server, 
                external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Airport Crowd Analysis'


# define main html layout for our application
app.layout = html.Div(
    [

        # simple narbar 
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Home", href="#")),
                dbc.DropdownMenu(
                    children=[
                        dbc.DropdownMenuItem("Support", header=True),
                        dbc.DropdownMenuItem("Support", href="#"),
                ],
                    nav=True,
                    in_navbar=True,
                    label="More",
                    ),
                ], brand="AirCrowd Platform", 
                   brand_href="#",
                   color="dark", 
                   dark=True
            ),

        # app title
        dbc.Row([
                html.Div([
                    html.H1("Airport Crowd Analysis Portal"),
                    dcc.Markdown(''' 
                        *Integrated crowd-monitoring and social-distancing analysis powered by pose estimation.*
                        '''),
                ], style={"width": "100%", 
                          "text-align": "center",
                          "padding-top" : 10,
                          "background-color": '#F8F9F9'})
            ]),

        # plot type selection - using a drop-down menu
        dbc.Row([

                dbc.Col([
                        html.H3("Select Type of Plot:", 
                                style={"width": "100%", 
                                       "text-align": "right"})
                    ], width=6),

                dbc.Col([
                        dcc.Dropdown(
                            id='plot-type-dropdown',
                            options=[{'label': i, 'value': i} for i in PLOT_RESULT_TYPES],
                            value=PLOT_RESULT_TYPES[0],
                            style={'height' : '60%', 'width':'100%', 
                                   'text-align' : 'left', 'background-color' : '#A5E8FF'})
                    ], width=2)
            ], style={'padding-top' : 20, "background-color": '#F8F9F9'}),

        # area selection region - using a drop-down menu
        dbc.Row([

                dbc.Col([
                        html.H3("Select Area:", 
                                style={"width": "100%", 
                                       "text-align": "right"}),
                    ], width=6),

                dbc.Col([
                        dcc.Dropdown(
                            id='area-dropdown',
                            options=[{'label': i, 'value': i} for i in AVAILABLE_AREAS],
                            value=AVAILABLE_AREAS[0],
                            style={'height' : '60%', 'width':'100%', 
                                   'text-align' : 'left', 'background-color' : '#82E0AA'})
                    ], width=2)
            ], style={'padding-top' : 20, "background-color": '#F8F9F9'}),

    # horizontal line to divide top section from results below
    dbc.Row(
        dbc.Col(
            html.Hr(style={'borderWidth': "1.0vh", "width": "100%", 
                           "backgroundColor": "#BBBBBB","opacity":"1"}),
                    width={'size':10, 'offset':1}),

            style={"background-color": '#F8F9F9'}), 


    # html content to be generated for selected area
    html.Div(children=[html.Div(id='video-feeds')])

    ])


@server.route('/static/<path:path>')
def serve_static(path):
    """ Serve item from static DIR from path given """ 
    return flask.send_from_directory(path)


@app.callback(
    Output('video-feeds', 'children'),
    Input('area-dropdown', 'value'),
    Input('plot-type-dropdown', 'value'))
def load_chosen_area(value, plot_type):
    """ Callback function to show video feed and associated results for the chosen
        area, according to the value of the dropdown menu (area-dropdown).

    Args:
        value (str) : image name (from dropdown menu) to display.
        plot_type (str) : type of results plot to show for chosen area.
    """ 
    # get location-specific directory
    area_dir = os.path.join(DATA_DIR, value)

    input_video = f"{value}.mov"
    results_video = f"{value}_{plot_type}.mov"

    # loads results / statistics for chosen area
    frame_results = pd.read_json(os.path.join(area_dir, 
                                 f"{value}_frame_results.JSON"))
    person_results = pd.read_json(os.path.join(area_dir, 
                                 f"{value}_person_results.JSON"))

    # reset index to use as col later
    frame_results = frame_results.reset_index()
    person_results = person_results.reset_index()

    # create stacked plot of mask proportions
    mask_fig = go.Figure()
    x_vals = frame_results.index.values
    mask_counts = frame_results['mask_count'].values
    person_counts = frame_results['person_count'].values

    # create total persons stacked plot with mask usage breakdown
    mask_fig.add_trace(go.Scatter(x=x_vals, y=mask_counts, marker={'color':"green"},
                        fill='tozeroy', name='Wearing Mask')) 
    mask_fig.add_trace(go.Scatter(x=x_vals, y=person_counts, marker={'color':"red"},
                        fill='tonexty', name="No Mask"))

    mask_fig.update_layout(title_text="Total Persons Mask Status",
                            title_font_size=25, 
                            title_x=0.5,
                            xaxis = dict(title = 'Frame (Timestep)'),
                            yaxis = dict(title = 'Count'), 
                            legend=dict(orientation="h", yanchor="bottom", 
                                        y=1.02, xanchor="right", x=1),
                            margin=dict(r=0))

    # create stacked plot of pose status
    pose_fig = go.Figure()
    standing_counts = frame_results['standing_count'].values

    pose_fig.add_trace(go.Scatter(x=x_vals, y=standing_counts, marker={'color':"blue"},
                        fill='tozeroy', name='Standing / Walking')) 
    pose_fig.add_trace(go.Scatter(x=x_vals, y=person_counts, marker={'color':"orange"},
                        fill='tonexty', name="Sitting / Other")) 

    pose_fig.update_layout(title_text="Total Persons Pose Status",
                            title_font_size=25,
                            title_x=0.5,
                            xaxis = dict(title = 'Frame (Timestep)'),
                            yaxis = dict(title = 'Count'), 
                            legend=dict(orientation="h", yanchor="bottom", 
                                        y=1.02, xanchor="right", x=1),
                            margin=dict(r=0))

    # create total risk plot
    risk_fig = go.Figure()
    risk_vals = frame_results['total_risk_profile'].values
    risk_fig.add_trace(go.Scatter(x=x_vals, y=risk_vals, mode='lines',
                       marker={'color':"red"})) 

    risk_fig.update_layout(title_text="Total Risk Profile",
                            title_font_size=25, 
                            title_x=0.5,
                            xaxis = dict(title = 'Frame (Timestep)'),
                            yaxis = dict(title = 'Risk Profile'))

    # create simple number of persons plot
    count_fig = go.Figure()
    count_fig.add_trace(go.Scatter(x=x_vals, y=person_counts, mode='lines', 
                        marker={'color':"blue"})) 

    count_fig.update_layout(title_text="Total Person Count",
                            title_font_size=25, 
                            title_x=0.5,
                            xaxis = dict(title = 'Frame (Timestep)'),
                            yaxis = dict(title = 'Person Count'))


    # create summary of proportions plot
    prop_fig = go.Figure()
    p1 = frame_results['social_distancing_compliance'].values
    p2 = frame_results['mask_proportions'].values
    p3 = frame_results['standing_proportions'].values
    prop_fig.add_trace(go.Scatter(x=x_vals, y=p1, mode='lines', opacity=.5,
                       marker={'color':'orange'}, name="Dist. Compliance")) 
    prop_fig.add_trace(go.Scatter(x=x_vals, y=p2, mode='lines', opacity=.5,
                       marker={'color':'green'}, name="Mask Usage"))
    prop_fig.add_trace(go.Scatter(x=x_vals, y=p3, mode='lines', opacity=.5,
                       marker={'color':'blue'}, name="Standing Proportion"))

    prop_fig.update_layout(title_text="Social Distancing Proportions",
                           title_font_size=25, 
                           title_x=0.5,
                           xaxis = dict(title = 'Frame (Timestep)'),
                           yaxis = dict(title = 'Proportion'),
                           legend=dict(orientation="h", yanchor="bottom", 
                                       y=1.02, xanchor="right", x=1),
                           margin=dict(r=0),)


    # create box plot of proportions (as above, but boxplot)
    prop_box_fig = go.Figure()
    prop_box_fig.add_trace(go.Box(y=p1, name='Dist. Compliance', line=dict(color='orange')))
    prop_box_fig.add_trace(go.Box(y=p2, name='Mask Usage', line=dict(color='green')))
    prop_box_fig.add_trace(go.Box(y=p3, name='Standing Proportion', line=dict(color='blue')))
    prop_box_fig.update_layout(title_text="Proportions Boxplots",
                               title_font_size=25, 
                               title_x=0.5,
                               yaxis = dict(title = 'Proportion'),
                               showlegend=False,
                               legend=dict(orientation="h", yanchor="bottom", 
                                           y=1.02, xanchor="right", x=1))


    # create dash table to display frame-level results
    frame_dt = dash_table.DataTable(
        id = 'frame-table', 
        data=frame_results.to_dict('records'),
        columns=[{"name": i, "id": i} for i in frame_results.columns], 
        page_size=10,
        filter_action = 'native',
        sort_action = 'native',
        sort_mode = 'multi',
        export_format="csv",

        style_cell={'whiteSpace': 'normal', 
                    'height': 'auto',
                    'textAlign': 'left'},

        style_table={'height': '400px', 'overflowY': 'auto'},

        style_data={'color': 'black', 
                    'backgroundColor': 'white'}, 

        style_data_conditional=[
            {'if': 
                {'row_index': 'odd'},
                 'backgroundColor': 'rgb(220, 220, 220)'}])


    # create dash table to display person-level results
    person_dt = dash_table.DataTable(
        id = 'person-table', 
        data=person_results.to_dict('records'),
        columns=[{"name": i, "id": i} for i in person_results.columns], 
        page_size=10,
        filter_action = 'native',
        sort_action = 'native',
        sort_mode = 'multi',
        export_format="csv",

        style_cell={'whiteSpace': 'normal', 
                    'height': 'auto',
                    'textAlign': 'left'},

        style_table={'height': '400px', 'overflowY': 'auto'},

        style_data={'color': 'black', 
                    'backgroundColor': 'white'},

        style_data_conditional=[
            {'if': 
                {'row_index': 'odd'},
                 'backgroundColor': 'rgb(220, 220, 220)'},
            {'if': 
                {'column_id': 'respect_social_distancing', 
                 'filter_query': '{respect_social_distancing} = 0'}, 
                 'backgroundColor': 'pink'},
            {'if':
                {'column_id': 'respect_social_distancing', 
                 'filter_query': '{respect_social_distancing} = 1'}, 
                 'backgroundColor': 'lightgreen'},
            {'if': 
                {'column_id': 'mask_preds', 
                 'filter_query': '{mask_preds} = 0'}, 
                 'backgroundColor': 'pink'},
            {'if':
                {'column_id': 'mask_preds', 
                 'filter_query': '{mask_preds} = 1'}, 
                 'backgroundColor': 'lightgreen'}])


    return_content = html.Div([

        # row to display input video and associated heatmap predictions
        dbc.Row(
            [
                dbc.Col([
                    html.H4(f"{value} Original Video", style={"width": "80%", "text-align": "center"}),
                    html.Video(src=f'/static/{value}/{input_video}', loop=True, autoPlay=True, 
                                style={'width' : '100%', 'padding': 1})
                    ], style={"background-color": '#ecfaff ', 'padding-top' : 20, 
                              'padding-bottom' : 20, "text-align": "center"}, 
                       width=4),
                dbc.Col([
                    html.H4(f"{value} {plot_type} Prediction", style={"width": "80%", "text-align": "center"}),
                    html.Video(src=f'/static/{value}/{results_video}', loop=True, autoPlay=True, 
                                style={'width' : '100%', 'padding': 1})
                    ], style={"background-color": '#f0fbf5 ', 'padding-top' : 20, 
                              'padding-bottom' : 20, "text-align": "center"}, 
                       width=4),
                dbc.Col([
                    dcc.Graph(id='mask-fig', figure=mask_fig, 
                              style={'height' : 400, 'width' : '100%'}),
                    dcc.Graph(id='risk-profile-fig', figure=risk_fig, 
                              style={'height' : 400, 'width' : '100%'}),
                    dcc.Graph(id='pose-fig', figure=pose_fig, 
                              style={'height' : 400, 'width' : '100%'}),
                    dcc.Graph(id='person-counts-fig', figure=count_fig, 
                              style={'height' : 400, 'width' : '100%'}),
                    dcc.Graph(id='proportion-fig', figure=prop_fig, 
                              style={'height' : 400, 'width' : '100%'}),
                    dcc.Graph(id='proportions-boxplot', figure=prop_box_fig, 
                              style={'height' : 400, 'width' : '100%'}),
                    ], className="h-100 overflow-scroll", style={"max-height": 400,"overflow": 'hidden', "overfow-y": 'scroll'}, width=4),
               
                
            ],
        ),

        ### section to display averaged risk density-heatmap results
        dbc.Row(
            dbc.Col(
                [
                    html.H4(f"{value} Averaged Risk Regions", 
                            style={"width": "100%", "text-align": "center"}),
                    dbc.Carousel(
                            items=[
                                {"key": "1", "src": f"/static/{value}/{value}_avg_heatmap_5.png",
                                "header": "Averaged Risk Regions", "caption" : "Past 5 seconds"},
                                {"key": "2", "src": f"/static/{value}/{value}_avg_heatmap_10.png",
                                "header": "Averaged Risk Regions ", "caption": "Past 10 seconds",},
                                {"key": "3", "src": f"/static/{value}/{value}_avg_heatmap_all.png",
                                "header": "Averaged Risk Regions ", "caption": "All timesteps"},
                        ],
                        controls=True,
                        indicators=False,
                    )

                ], width={'size':6, 'offset':3}

            ), style={"padding-top":30, 'background-color' : '#F8F9F9'}
        ),


        # horizontal divider to nicely seperate content
        dbc.Row(
            dbc.Col(
                html.Hr(style={'borderWidth': "1.0vh", "width": "100%", 
                           "backgroundColor": "#BBBBBB","opacity":"1"}),
                        width={'size':12, 'offset':0})),

        dbc.Row(
            [
                dbc.Col([
                    html.H3(f"Scene Statistics and Results for {value}", 
                            style={"width": "100%", "text-align": "center"}),
                    ], width=12),
            ], style={'padding-top' : 20}),

        dbc.Row(
            [
                dbc.Col([
                    dcc.Graph(id='mask-fig', figure=mask_fig, 
                              style={'height' : 400, 'width' : '100%'}),
                    ], width=7),

                dbc.Col([
                    dcc.Graph(id='risk-profile-fig', figure=risk_fig, 
                              style={'height' : 400, 'width' : '100%'}),
                    ], width=5),
            ]),

        dbc.Row(
            [
                dbc.Col([
                    dcc.Graph(id='pose-fig', figure=pose_fig, 
                              style={'height' : 400, 'width' : '100%'}),
                    ], width=7),

                dbc.Col([
                    dcc.Graph(id='person-counts-fig', figure=count_fig, 
                              style={'height' : 400, 'width' : '100%'}),
                    ], width=5),
            ]),

        dbc.Row(
            [
                dbc.Col([
                    dcc.Graph(id='proportion-fig', figure=prop_fig, 
                              style={'height' : 400, 'width' : '100%'}),
                    ], width=7),

                dbc.Col([
                    dcc.Graph(id='proportions-boxplot', figure=prop_box_fig, 
                              style={'height' : 400, 'width' : '100%'}),
                    ], width=5),
            ]),

        
        # section divider before displaying frame and person results tables
        dbc.Row(
            dbc.Col([
                html.Hr(style={'borderWidth': "1.0vh", "width": "100%", 
                               "backgroundColor": "#BBBBBB","opacity":"1"}),
                html.H2(f"Data and Results for {value}", 
                            style={"width": "100%", "text-align": "center"})],
                        width={'size':12, 'offset':0}),
            style={'padding-bottom' : 10}), 

        # row to display frame-level results
        dbc.Row(
            [
                dbc.Col([
                    html.H4("Frame-Level Data", 
                            style={"width": "100%", "text-align": "center"}),
                    frame_dt
                    ], width={"size": 10, "offset": 1}),
            ]),

        # row to display person-level results
        dbc.Row(
            [
                dbc.Col([
                    html.H4("Individual Person-Level Data", 
                            style={"width": "100%", "text-align": "center"}),
                    person_dt
                    ], width={"size": 10, "offset": 1})
            ])
    ])

    return return_content


if __name__ == '__main__':
    app.run_server(debug=True)