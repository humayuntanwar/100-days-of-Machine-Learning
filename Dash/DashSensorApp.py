#Imports
import dash
import dash_core_components as dcc
import dash_html_components as html
import time
from collections import deque
import plotly.graph_objs as go
import random

#app name
app = dash.Dash('vehicle-data')

#our data
#simulating continious recieving data from sensors , with our dummy random data with deque
max_length = 200
times = deque(maxlen=max_length)
oil_temps = deque(maxlen=max_length)
intake_temps = deque(maxlen=max_length)
coolant_temps = deque(maxlen=max_length)
rpms = deque(maxlen=max_length)
speeds = deque(maxlen=max_length)
throttle_pos = deque(maxlen=max_length)

#creating a dictionary, key->value
#we will use a drop down allow user to select keys and then show mapped values
data_dict = {"Oil Temperature":oil_temps,
"Intake Temperature": intake_temps,
"Coolant Temperature": coolant_temps,
"RPM":rpms,
"Speed":speeds,
"Throttle Position":throttle_pos}


# main method
# takes all args as params
def update_obd_values(times, oil_temps, intake_temps, coolant_temps, rpms, speeds, throttle_pos):

    times.append(time.time())
    if len(times) == 1:
        #starting relevant values
        oil_temps.append(random.randrange(180,230))
        intake_temps.append(random.randrange(95,115))
        coolant_temps.append(random.randrange(170,220))
        rpms.append(random.randrange(1000,9500))
        speeds.append(random.randrange(30,140))
        throttle_pos.append(random.randrange(10,90))
    else:
        for data_of_interest in [oil_temps, intake_temps, coolant_temps, rpms, speeds, throttle_pos]:
            data_of_interest.append(data_of_interest[-1]+data_of_interest[-1]*random.uniform(-0.0001,0.0001))

    return times, oil_temps, intake_temps, coolant_temps, rpms, speeds, throttle_pos



times, oil_temps, intake_temps, coolant_temps, rpms, speeds, throttle_pos = update_obd_values(times, oil_temps, intake_temps, coolant_temps, rpms, speeds, throttle_pos)


#app layout 
app.layout = html.Div([
    html.Div([
        #floating title of app on left
        html.H2('Vehicle Data',
                style={'float': 'left',
                    }),
        ]),
        #creating dropdown menus
    dcc.Dropdown(id='vehicle-data-name',
                #labels and values
                options=[{'label': s, 'value': s}
                        for s in data_dict.keys()],
                #by defualt we will have 3 graphs pop up
                value=['Coolant Temperature','Oil Temperature','Intake Temperature'],
                multi=True
                ),
    #all divs using meterilized css
    html.Div(children=html.Div(id='graphs'), className='row'),
    #update live
    dcc.Interval(
        id='graph-update',
        interval=100),
    ], className="container",style={'width':'98%','margin-left':10,'margin-right':10,'max-width':50000})

#decorator, output, input and event
@app.callback(
    dash.dependencies.Output('graphs','children'),
    [dash.dependencies.Input('vehicle-data-name', 'value')],
    events=[dash.dependencies.Event('graph-update', 'interval')]
    )

#update graph function
def update_graph(data_names):
    #list of graphs
    graphs = []
    update_obd_values(times, oil_temps, intake_temps, coolant_temps, rpms, speeds, throttle_pos)
    #screen size and graphhs setting
    if len(data_names)>2:
        #small screen many graph one per row
        class_choice = 'col s12 m6 l4'
    elif len(data_names) == 2:
        #two data names handle
        class_choice = 'col s12 m6 l6'
    else:
        #for one thing
        class_choice = 'col s12'
    #iterate thorugh data names
    #give, name, fill and color
    for data_name in data_names:

        data = go.Scatter(
            x=list(times),
            y=list(data_dict[data_name]),
            name='Scatter',
            fill="tozeroy",
            fillcolor="#6897bb"
            )
        #append to list and build graph element itself
        graphs.append(html.Div(dcc.Graph(
            id=data_name,
            animate=True,
            figure={'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(times),max(times)]),
                                                        yaxis=dict(range=[min(data_dict[data_name]),max(data_dict[data_name])]),
                                                        #trying to get on screen as possible
                                                        #for charts
                                                        margin={'l':50,'r':1,'t':45,'b':1},
                                                        #title text
                                                        title='{}'.format(data_name))}
            ), className=class_choice))

    return graphs

#include external css and js
external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
for css in external_css:
    app.css.append_css({"external_url": css})
#js
external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js']
for js in external_css:
    app.scripts.append_script({'external_url': js})



#run app
if __name__ == '__main__':
    app.run_server(debug=True)