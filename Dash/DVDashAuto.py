
import dash
#event auto update
from dash.dependencies import Output, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
#set axis limits for charts. 
import plotly.graph_objs as go
# set a size limit (maxlen)
from collections import deque

#sample data
X = deque(maxlen=20)
X.append(1)
Y = deque(maxlen=20)
Y.append(1)

#appname start
app = dash.Dash(__name__)

#layout styling
app.layout = html.Div(
    [
        dcc.Graph(id='live-graph', animate=True),
        dcc.Interval(
            id='graph-update',
            #set pdate interval
            interval=1*1000
        ),
    ]
)

#call backs for event handling
@app.callback(Output('live-graph', 'figure'),
            events=[Event('graph-update', 'interval')])


#auto update grapgh method
def update_graph_scatter():
    X.append(X[-1]+1)
    Y.append(Y[-1]+Y[-1]*random.uniform(-0.1,0.1))

    data = plotly.graph_objs.Scatter(
            x=list(X),
            y=list(Y),
            name='Scatter',
            mode= 'lines+markers'
            )

    return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                yaxis=dict(range=[min(Y),max(Y)]),)}



#run app
if __name__ == '__main__':
    app.run_server(debug=True)