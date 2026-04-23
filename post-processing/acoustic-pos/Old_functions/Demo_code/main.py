"""Example of AI multitask operation."""
import numpy as np
import matplotlib
import json
import localFunctions as lf
from scipy import signal
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import Dash, html, dcc, Output, Input
import dash_bootstrap_components as dbc
from collections import deque
from dash_bootstrap_templates import load_figure_template
import base64
import asyncio
import time
import webbrowser
import warnings
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
#matplotlib.use('TkAgg')

# #############################################################################################
# ----------------------------------- Configurations ------------------------------------------
# #############################################################################################
with open('config.json') as json_file:
    config = json.load(json_file)

# Configuration file information
# "chirp_f_start": Start frequency of the chrip signal
# "chirp_f_stop": Stop frequency of the chirp signal
# "chirp_duration": Duration of the signal in s
# "chirp_DC": DC offset of the chirp
# "chirp_ampl": Amplitude of chirp signal in V
# "wake_up_duration": Duration of wake up for the microphone
# "max_val_out": Max. output voltage of daq
# "min_val_out": Min. output voltage of daq
# "max_val_in": Max. output voltage of daq
# "min_val_in": Min. output voltage of daq
# "sample_rate": Sample rate to create the signal, for speaker and microphone
# "n_meas": Amount of chrips sended for 1 ranging estimation
# "peak_prominence_factor": peak prominence value max 1
# "temperature": temperature in celcius to determine speed op sound
# "interval": time in between positioning estimates
# "plot_signals": Plot signals true or false
# "plot_spectrogram": Plot spectrogram true or false
# "plot_pulse_compr": Plot pulse compression true or false

# #############################################################################################
# ---------------------------------------- SETUP ---------------------------------------------
# #############################################################################################
fs = config["sample_rate"]

anchor_name = []
anchor_name = lf.add_all(anchor_name)
anchor_name = lf.remove_tile(anchor_name, ['G02'])
anchor_name = lf.clean_redundant(anchor_name)

# add_segment(A, ...)
# add_part_segment(A, North)
# add_surface(East, West or Roof)

#      W
#   _______
# S | Roof | N
#   -------
#      E

# add_dam_pattern()
# add_all()
# clean_redundant(anchor_list)
# remove_tile(anchor_list, ['A01', 'A03'])

anchor_xyz = lf.create_anchor_position_matrix(anchor_name)

room_dim = config["room_dim"]
height = room_dim[2]
vertices = np.array([[0, 0], [0, room_dim[1]], [room_dim[0], room_dim[1]], [room_dim[0], 0]])

# Create chirp signal
chirp_transmit = lf.create_chirp(start_freq=config["chirp_f_start"], stop_freq=config["chirp_f_stop"],
                        chirp_duration=config["chirp_duration"], fs=fs,
                        amplitude=config["chirp_ampl"], offset=config["chirp_DC"])

# lf.plot_spectrogram('Spectogram TX Chirp', chirp, fs, config["chirp_duration"], 0)

external_stylesheets =[dbc.themes.SLATE] #SLATE
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#FFFFFF'
}

kuleuven_logo = 'kuleuven_logo.png'
dramco_logo = 'dramco_logo.png'
kuleuven_logo_base64 = base64.b64encode(open(kuleuven_logo, 'rb').read()).decode('ascii')
dramco_logo_base64 = base64.b64encode(open(dramco_logo, 'rb').read()).decode('ascii')


app.layout =  html.Div([
                    dbc.Container(children=[
                        dbc.Row([
                            dbc.Col(html.Div(children=[html.H1(children='Ultrasound Indoor Positioning',
                                                                style={ 'textAlign': 'left',
                                                                'color': colors['text']}
                                                      ),
                                                      html.Div(children='Daan Delabie',
                                                                style={ 'textAlign': 'left',
                                                                        'color': colors['text']
                                                                }
                                                      )
                                    ], style={'margin-right':'15px', 'margin-top':'15px'}), width=6
                            ),
                            dbc.Col(html.Img(src='data:image/png;base64,{}'.format(kuleuven_logo_base64),
                                             style={'height': '50%', 'width': '50%', 'margin': '5px', 'margin-top':'15px'}), width=2),
                            dbc.Col(html.Img(src='data:image/png;base64,{}'.format(dramco_logo_base64),
                                             style={'height': '50%', 'width': '50%', 'margin': '5px', 'margin-top':'15px'}), width=1),
                        ], justify='start')]),

                    html.Div([dbc.Row([
                                dbc.Col([
                                    dbc.Row(html.Div(children=[
                                        html.H5(children='Estimated Position',
                                                style={'textAlign': 'left', 'color': colors['text']}),
                                        html.B(children='X: ',
                                               style={'textAlign': 'left', 'color': colors['text'],
                                                      'display': 'inline'}),
                                        html.Div(children='X', id='X_val',
                                                 style={'textAlign': 'center', 'color': colors['text'],
                                                        'display': 'inline'}),
                                        html.Div(children=' m', style={'textAlign': 'left', 'color': colors['text'],
                                                                       'display': 'inline', 'margin-right': '10px'}),
                                        html.B(children=' Y: ', style={'textAlign': 'left', 'color': colors['text'],
                                                                       'display': 'inline'}),
                                        html.Div(children='Y', id='Y_val',
                                                 style={'textAlign': 'center', 'color': colors['text'],
                                                        'display': 'inline'}),
                                        html.Div(children=' m', style={'textAlign': 'left', 'color': colors['text'],
                                                                       'display': 'inline', 'margin-right': '10px'}),
                                        html.B(children=' Z: ', style={'textAlign': 'left', 'color': colors['text'],
                                                                       'display': 'inline'}),
                                        html.Div(children='Z', id='Z_val',
                                                 style={'textAlign': 'center', 'color': colors['text'],
                                                        'display': 'inline'}),
                                        html.Div(children=' m', style={'textAlign': 'left', 'color': colors['text'],
                                                                       'display': 'inline'}),
                                        dcc.Interval(id='graph-update-pos', interval=config["interval"] * 1,
                                                     n_intervals=0, disabled=False),
                                    ], style={'margin-top': '100px', 'margin-left': '200px'})),   #25

                                    dbc.Row([
                                        html.Div([
                                            dcc.Graph(id='3Dfig', animate=True, animation_options={"frame":{"redraw":True}}),
                                            dcc.Interval(id='graph-update', interval=config["interval"] * 1, n_intervals=0, disabled=False),
                                        ])
                                    ],),

                                ], width=6),
                                dbc.Col([
                                    dbc.Row([
                                        dcc.Graph(id='rawgraph', animate=True),
                                        dcc.Interval(id='graph-update-raw', interval=config["interval"] * 1, n_intervals=0, disabled=False),
                                    ]),
                                    dbc.Row([
                                        dcc.Graph(id='corr-graph', animate=True),
                                        dcc.Interval(id='graph-update-corr', interval=config["interval"] * 1, n_intervals=0, disabled=False),
                                    ])
                                ], width=6),
                            ]),
                    ]),
            ])

@app.callback(
    [Output('3Dfig', 'figure'), Output('rawgraph', 'figure'), Output('corr-graph', 'figure'), Output('X_val', 'children'), Output('Y_val', 'children'), Output('Z_val', 'children')],
    [Input('graph-update', 'n_intervals')]
)

def update_figure(n):
    # Run the DAQ process
    RX_data = lf.DAQ(chirp_transmit, anchor_name)
    # lf.plot_signals(RX_data, chirp)
    received_data = np.array(RX_data)

    # downsample_freq
    chirp = lf.resample_signals(config['sample_rate'], config['downsample_freq'], chirp_transmit)
    received_data = lf.resample_signals(config['sample_rate'], config['downsample_freq'], received_data.T).T

    # HARDWARE CHECK IF MICROPHONE IS NOT POWERED
    mean_vals_mics = np.mean(received_data, axis=1)
    idx_broken_mic = np.where(mean_vals_mics < 1.5)[0]
    if idx_broken_mic.size != 0:
        print('!!! HARDWARE ERROR !!! BROKEN or NOT POWERED MICROPHONE(S): ', [anchor_name[i] for i in idx_broken_mic])
        lf.plot_signals(received_data, chirp, anchor_name)
        exit()

    if config['HPF']:
        # apply filter to get only ultrasound
        received_data = lf.butter_highpass_filter(received_data, 15000, config['downsample_freq'], 10)

    if config['filter_rms']:
        print('Anchor names: \n', anchor_name)
        # Create mask
        filter_mask = lf.get_filter_mask(received_data, config['filter_threshold'])

        # Filter data based on mask
        deleted_data, received_data = received_data[~filter_mask], received_data[filter_mask]
        used_anchor_xyz, removed_anchor_xyz = anchor_xyz[filter_mask], anchor_xyz[~filter_mask]
        used_anchors, removed_anchors = lf.filter_anchors(anchor_name, filter_mask)

    else:
        used_anchors = anchor_name
        removed_anchors = []
        used_anchor_xyz = anchor_xyz
        removed_anchor_xyz = []
        deleted_data = []

    # print('Anchors used: ', used_anchors)
    print('Anchors removed: ', removed_anchors)

    if config['anchor_to_plot'] == "all":
        rawgraph = lf.plot_signals_dash(received_data, chirp, config["downsample_freq"], used_anchors)
        rawgraph_removed = lf.plot_signals_dash(deleted_data, chirp, config["downsample_freq"], removed_anchors)

    else:
        # Check if anchor is in used or removed list and get index
        if config['anchor_to_plot'] in used_anchors:
            index_anchor = used_anchors.index(config['anchor_to_plot'])
            data_to_plot = received_data[index_anchor, :]

        elif config['anchor_to_plot'] in removed_anchors:
            index_anchor = removed_anchors.index(config['anchor_to_plot'])
            data_to_plot = deleted_data[index_anchor, :]

        else:
            print('no anchor found to make a plot')

        # Make plots based on previous information
        rawgraph = lf.plot_signals_dash(data_to_plot, chirp, config["downsample_freq"], config['anchor_to_plot'])

    if config["method"]  == 'ToA':
        # # -----------------------
        # #           ToA
        # # -----------------------
        # # Determine ToA ranges (with PP)
        distances_meas, pulse_compr_all, LPF_all = lf.calc_ranges_ToA(received_data, chirp)
        #print('\nMeasured distances (ToA) between mic and speaker (in m) \n', distances_meas)

        if config['anchor_to_plot'] == "all":
            corr_graph = lf.plot_pulsecomp_dash_ToA(pulse_compr_all, LPF_all, distances_meas, config["downsample_freq"], used_anchors, chirp)

        else:
            # Check if anchor is in used or removed list and get index
            if config['anchor_to_plot'] in used_anchors:
                index_anchor = used_anchors.index(config['anchor_to_plot'])
                data_to_plot_pulse_compr = pulse_compr_all[index_anchor, :]
                LPF_to_plot = LPF_all[index_anchor, :]
                distances_meas_plot = distances_meas[index_anchor]

            elif config['anchor_to_plot'] in removed_anchors:
                index_anchor = removed_anchors.index(config['anchor_to_plot'])
                distances_meas_del, pulse_compr_all_del, LPF_all_del = lf.calc_ranges_ToA(deleted_data, chirp)
                data_to_plot_pulse_compr = pulse_compr_all_del[index_anchor, :]
                LPF_to_plot = LPF_all_del[index_anchor, :]
                distances_meas_plot = distances_meas_del[index_anchor]

            else:
                print('no anchor found to make a plot')

            corr_graph = lf.plot_pulsecomp_dash_ToA(data_to_plot_pulse_compr, LPF_to_plot, distances_meas_plot, config["downsample_freq"],
                                                    config['anchor_to_plot'], chirp)

        # # Determine position estimate based on the ToA ranges (LS)
        position_estimate = lf.LS_positioning(used_anchor_xyz, distances_meas.flatten(), np.array([4.0, 2.0, 1.5]))
        print('\n ToA Estimated position: \n', position_estimate)
        # TODO if want both TDoA and ToA: could reuse index estimates from ToA


    elif config["method"]  == 'TDoA':
        # -----------------------
        #           TDoA
        # -----------------------
        # Determine the TDoA times (with PP) (calculate all delays to anchors) and position with LS
        position_estimate, pulse_compr_all, LPF_all, corr_index_array = lf.calc_3dpos_TDoA(used_anchor_xyz, received_data, chirp)
        print('\n TDoA Estimated position: \n', position_estimate)
        #TODO fix flipped data + add only one anchor thing like for ToA
        corr_graph = lf.plot_pulsecomp_dash_TDoA(pulse_compr_all, LPF_all, corr_index_array, used_anchors)

    else:
        print('NO SELECTED POSITIONING METHOD')
        position_estimate = np.array([0,0,0])

    fig_3D = lf.plot_generated_anchors(vertices, height, used_anchor_xyz, removed_anchor_xyz, position_estimate, '3Dfig', 'Estimated Position', used_anchors, removed_anchors)
    X = position_estimate[0]
    Y = position_estimate[1]
    Z = position_estimate[2]

    return [fig_3D, rawgraph, corr_graph, np.round(X,3), np.round(Y,3), np.round(Z,3)]

if __name__ == "__main__":
    # Setup synchronization of the DAQ
    lf.DAQ_set_sync()

    port = 8050
    webbrowser.open("http://localhost:{}".format(port), new=0, autoraise=True)
    app.run_server()

    # clean-up synchronization of the DAQ
    #lf.DAQ_reset_sync()
