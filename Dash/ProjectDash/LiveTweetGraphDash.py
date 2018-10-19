#using code from our DVDASHAUTO
import dash
#event auto update
from dash.dependencies import Input, Output, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
#set axis limits for charts. 
import plotly.graph_objs as go
# set a size limit (maxlen)
from collections import deque

import sqlite3
import pandas as pd

#appname start
app = dash.Dash(__name__)

#layout styling
app.layout = html.Div(
    [   html.H2('Live Twitter Sentiment'),
        dcc.Input(id='sentiment_term', value='twitter', type='text'),
        dcc.Graph(id='live-graph', animate=True),
        dcc.Interval(
            id='graph-update',
            interval=1*1000
        ),
    ]
)

#call backs for event handling
@app.callback(Output('live-graph', 'figure'),
            [Input(component_id='sentiment_term',component_property='value')],
            events=[Event('graph-update', 'interval')])


#auto update grapgh method
def update_graph_scatter(sentiment_term):
    try: 
        #connect to db create cursor
        conn = sqlite3.connect('twitter.db')
        c = conn.cursor()

        #build the data frame 
        #pass query and conn
        df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE ? ORDER BY unix DESC LIMIT 1000", conn ,params=('%' + sentiment_term + '%',))
        # sort by unix
        df.sort_values('unix', inplace=True)
        # column names 
        df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df)/5)).mean()
        df.dropna(inplace=True)

        #create our xs and ys upto 100, last 100 vals , x is unix, y is sentiment smoothed
        X = df.unix.values[-100:]
        Y = df.sentiment_smoothed.values[-100:]


        data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Scatter',
                mode= 'lines+markers'
                )

        return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                    yaxis=dict(range=[min(Y),max(Y)]),)}
    #  catch error using exception write to file
    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')


#run app
if __name__ == '__main__':
    app.run_server(debug=True)