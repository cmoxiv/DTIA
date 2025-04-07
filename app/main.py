
from bokeh.io import show
from bokeh.layouts import column
from bokeh.models import FileInput, Button
from bokeh.plotting import figure, curdoc

file_input = FileInput()

# add a button widget and configure with the call back
button = Button(label="Press Me")
button.on_click(lambda: True)




from datetime import date
from random import randint

from bokeh.io import show
from bokeh.models import ColumnDataSource, DataTable, DateFormatter, TableColumn

data = dict(
        dates=[date(2014, 3, i+1) for i in range(10)],
        downloads=[randint(0, 100) for i in range(10)],
    )
source = ColumnDataSource(data)

columns = [
        TableColumn(field="dates", title="Date", formatter=DateFormatter()),
        TableColumn(field="downloads", title="Downloads"),
    ]
data_table = DataTable(source=source, columns=columns, width=400, height=280)

# show(data_table)


curdoc().add_root(column(data_table, file_input, button))
