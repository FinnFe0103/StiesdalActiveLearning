from dash import Dash
from dash_bootstrap_components.themes import BOOTSTRAP

from data import import_data
from layout import create_layout

sensors, caselist, sim_results = import_data()
app = Dash(__name__, external_stylesheets=[BOOTSTRAP])
app.title = "Active Learning Selection Algorithm"
app.layout = create_layout(app, sensors)

server = app.server

if __name__ == '__main__':
    app.run(debug=True)

# @callback(
#     Output('graph-content', 'figure'),
#     Input('dropdown-selection', 'value')
# )
# def update_graph(value):
#     dff = df[df.country==value]
#     return px.line(dff, x='year', y='pop')