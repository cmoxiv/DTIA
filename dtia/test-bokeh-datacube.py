from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import Button
from bokeh.models import ColumnDataSource, DataCube, GroupingInfo
from bokeh.models import StringFormatter, SumAggregator, TableColumn
from bokeh.models import AvgAggregator

from bokeh.plotting import figure, curdoc

source = ColumnDataSource(data=dict(
    d0=['A', 'E', 'E', 'E', 'A', 'A', 'M'],
    d1=['B', 'D', 'D', 'H', 'K', 'L', 'N'],
    d2=['C', 'F', 'G', 'H', 'K', 'L', 'O'],
    px=[10, 20, 30, 40, 50, 60, 70],
))

target = ColumnDataSource(data=dict(row_indices=[], labels=[]))

formatter = StringFormatter(font_style='bold')

columns = [
    TableColumn(field='d2', title='Name', width=80, sortable=False, formatter=formatter),
    TableColumn(field='px', title='Price', width=40, sortable=False),
]

grouping = [
    GroupingInfo(getter='d0', aggregators=[AvgAggregator(field_='px')]),
    GroupingInfo(getter='d1', aggregators=[AvgAggregator(field_='px')]),
]

# add a button widget and configure with the call back
# button = Button(label="Press Me")
# button.on_click(lambda: True)

cube = DataCube(source=source, columns=columns, grouping=grouping, target=target)

# p = figure(x_range=(0, 100), y_range=(0, 100), toolbar_location=None)

from bokeh.models import FileInput
file_input = FileInput()

# curdoc().add_root(column(cube, button))
# curdoc().add_root(column(cube, file_input))
curdoc().add_root(file_input)




# show(cube)
