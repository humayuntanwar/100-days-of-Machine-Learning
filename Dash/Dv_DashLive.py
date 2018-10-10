# getting live data and producing graphs
import pandas_datareader as web
import datetime
# for data
import quandl 
#read keey

# main import
import dash
#dash components
import dash_core_components as dcc
#dash html components
import dash_html_components as html
#lets get user involved
from dash.dependencies import Input, Output


apikey = open('key.txt','r').read()

#start app
app = dash.Dash()
#stock
stock = 'WIKI/GOOGL'
#start end dates

#print(df.head())
#df = web.DataReader("TSLA", 'quandl', start, end)
#df.reset_index(inplace=True)
#df.set_index("Date", inplace=True)
#df = df.drop("Symbol", axis=1)
#print(df.head())

app.layout = html.Div(children=[
                    
                    html.Div(children='''
                    Symbol to graph'''),
                    #input field
                    dcc.Input(id='input', value='', type='text'),
                    #output field
                    html.Div(id='output-graph'),
                    ])
#That function that will do that will need to be wrapped, 
@app.callback(
    Output(component_id='output-graph', component_property='children'),
    [Input(component_id='input', component_property='value')]
)

#update graphing
def update_value(input_data):
    start = datetime.datetime(2015,1,1)
    end = datetime.datetime.now()
    #get data
    # if input_data == '': 
    #     return None
    # elif input_data not in tickers: 
    #     return '{} not a valid ticker'.format(input_data)
    # else:
    #df=quandl.get(input_data, authtoken=api_key, start_date=start,end_date=end)
    df = web.DataReader('WIKI/'+input_data,'quandl',start, end,access_key=apikey)
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    #df = df.drop("Symbol", axis=1)
        #return graph
    return dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': df.index, 'y': df.Close, 'type': 'line', 'name': input_data},
            ],
            'layout': {
                'title': input_data
            }
        }
    )
#run app
# keep debug tru for debugging
if __name__ == '__main__':
    app.run_server(debug=True)
