# pip install dash dash-renderer dash-html-components dash-core-components poltly
# main import
import dash
#dash components
import dash_core_components as dcc
#dash html components
import dash_html_components as html
#lets get user involved
from dash.dependencies import Input, Output
#start app
app = dash.Dash()

#app layout, html of our project
# the div contains everything
#we will use labels, childern
app.layout = html.Div(children=[
    #input field
    dcc.Input(id='input',value='enter something,,',type='text'),
    #output div
    html.Div(id='output')
                    ])

#wrapper
@app.callback(
    Output(component_id='output',component_property='children'),
    [Input(component_id='input',component_property='value')]
    )

def update_value(input_data):
    try:
        #basic logic to square the given number
        return str(float(input_data)**2)
    except:
        #handle exception
        return 'some error'
#run app
# keep debug tru for debugging
if __name__ == '__main__':
    app.run_server(debug=True)
