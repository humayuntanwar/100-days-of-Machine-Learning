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
#creating two fugures which are graphs with dummy data, type of graph and name of grapgh
# giving our layout a title
app.layout = html.Div(children=[html.H1('Dash tutorials'),
                    dcc.Graph(id='example',figure={
                        'data':[
                            {'x':[1,2,3,4,5],'y':[5,6,7,2,1],'type':'line','name':'boats'},
                            {'x':[1,2,3,4,5],'y':[8,3,2,3,5],'type':'bar','name':'cars'},
                            ],
                            'layout':{
                                'title':'basic dash example'
                            }
                    })])

#run app
# keep debug tru for debugging
if __name__ == '__main__':
    app.run_server(debug=True)
