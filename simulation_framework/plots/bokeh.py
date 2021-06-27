from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.models import RangeTool
from bokeh.models import BoxSelectTool
from bokeh.models import LassoSelectTool
from bokeh.models import WheelZoomTool
from bokeh.models import BoxZoomTool
from bokeh.models import ResetTool
from bokeh.models import ColumnDataSource
import numpy as np
from scipy.stats.kde import gaussian_kde
from bokeh.models import Slider
import pickle
from bokeh.layouts import row, column


def range_plot(data):
    var_list = data.columns
    data_b = data[data['decision'] == "b"]
    data_s = data[data['decision'] == "s"]
    cds_b = ColumnDataSource(data={
            var_list[0]: data_b[var_list[0]],
            var_list[1]: data_b[var_list[1]],
            var_list[2]: data_b[var_list[2]],
            var_list[3]: data_b[var_list[3]],
            var_list[4]: data_b[var_list[4]],
            var_list[5]: data_b[var_list[5]],
            var_list[6]: data_b[var_list[6]],
            var_list[7]: data_b[var_list[7]],
            var_list[8]: data_b[var_list[8]]})

    cds_s = ColumnDataSource(data={
            var_list[0]: data_s[var_list[0]],
            var_list[1]: data_s[var_list[1]],
            var_list[2]: data_s[var_list[2]],
            var_list[3]: data_s[var_list[3]],
            var_list[4]: data_s[var_list[4]],
            var_list[5]: data_s[var_list[5]],
            var_list[6]: data_s[var_list[6]],
            var_list[7]: data_s[var_list[7]],
            var_list[8]: data_s[var_list[8]]})


    end = 1000
    p_1 = figure(title='Investment decisions',
                 x_axis_location="above",
                 x_axis_label='Number of Decisions',
                 y_axis_label="Ornstein-Uhlenbeck process",
                 x_range=(0, end),
                 y_range=(min(data["price"].values), max(data["price"].values)),
                 plot_height=300,
                 plot_width=900,
                 tools=[HoverTool(tooltips=[
                     ('OU-process', '@price'),
                     ('Bucket', '@bucket'),
                     ('Last Switch', '@last_switch'),
                     ('State', '@states'),
                     ('Greedy', '@greedy'),
                     ('Decision', '@decision'),
                     ('Epsilon', '@epsilon'),
                     ('Episode', '@episode')])], toolbar_location='below',
                 toolbar_sticky=False)
    p_1.add_tools(LassoSelectTool())
    p_1.add_tools(WheelZoomTool())
    p_1.add_tools(BoxZoomTool())
    p_1.add_tools(ResetTool())
    p_1.add_tools(BoxSelectTool())



    p_1.circle(x='time', y='price', alpha=0.6, source=cds_b,
               color='green', legend_label="Short to Long (Buy)")
    p_1.circle(x='time', y='price', alpha=0.6, source=cds_s,
               color='red', legend_label="Long to Short (Sell)")

    select = figure(title="Range Determination",
                    x_axis_type="linear",
                    plot_height=130, plot_width=900, y_range=p_1.y_range,
                    y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")

    range_tool = RangeTool(x_range=p_1.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select.circle(x='time', y='price', alpha=0.1, source=cds_b,
                  color='green')
    select.circle(x='time', y='price', alpha=0.1, source=cds_s,
                  color='red')

    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool
    p_1.legend.location = 'center_left'
    return p_1, select

def hist_and_kde(data, steps=10, bins=100):
    p_v = figure(title='Dynamic distribution of decisions',
                 x_axis_label="Ornstein-Uhlenbeck process",
                 y_axis_label="Density",
                 x_range=(min(data["price"].values), max(data["price"].values)),
                 plot_height=400,
                 plot_width=900, toolbar_location='below',
                 toolbar_sticky=False)

    dps = int(len(data) / steps)
    cds_b = ColumnDataSource(data={
        "top": [],
        "bottom": [],
        "left": [],
        "right": [],
        "x": [],
        "y": []})

    cds_s = ColumnDataSource(data={
        "top": [],
        "bottom": [],
        "left": [],
        "right": [],
        "x": [],
        "y": []})

    p_v.quad(top='top', bottom=0, left='left', right='right', source=cds_b,
             fill_color='green', line_color="white", alpha=0.4)

    p_v.quad(top='top', bottom=0, left='left', right='right', source=cds_s,
             fill_color='red', line_color="white", alpha=0.4)

    p_v.line(x='x', y='y', source=cds_b, line_color='green', line_width=2, alpha=1.0,
             legend_label="Short to Long (Buy)")
    p_v.line(x='x', y='y', source=cds_s, line_color='red', line_width=2, alpha=1.0, legend_label="Long to Short (Sell)")

    def update_plots(attr, old, new):
        lower = (slider_1.value - 1) * dps
        upper = slider_1.value * dps

        data_temp = data.iloc[lower:upper, :]
        data_temp_b = data_temp[data_temp['decision'] == "b"]
        data_temp_s = data_temp[data_temp['decision'] == "s"]

        hist_b, edges_b = np.histogram(data_temp_b['price'].values, density=True, bins=bins)
        hist_s, edges_s = np.histogram(data_temp_s['price'].values, density=True, bins=bins)

        pdf_b = gaussian_kde(data_temp_b['price'].values)
        pdf_s = gaussian_kde(data_temp_s['price'].values)

        x_b = np.linspace(min(data_temp_b['price'].values), max(data_temp_b['price'].values), bins)
        x_s = np.linspace(min(data_temp_s['price'].values), max(data_temp_s['price'].values), bins)

        new_data_b = {
            "top": hist_b,
            "bottom": np.zeros(len(hist_b)),
            "left": edges_b[:-1],
            "right": edges_b[1:],
            "x": x_b,
            "y": pdf_b(x_b)}

        new_data_s = {
            "top": hist_s,
            "bottom": np.zeros(len(hist_s)),
            "left": edges_s[:-1],
            "right": edges_s[1:],
            "x": x_s,
            "y": pdf_s(x_s)}

        cds_b.data = new_data_b
        cds_s.data = new_data_s

    slider_1 = Slider(start=1, end=steps, step=1, value=1, title='Slider')
    slider_1.on_change('value', update_plots)

    return p_v, slider_1

def hist_buckets(data, steps=10):
    p_v_1 = figure(title='Frequency of decisions per bucket',
                 x_axis_label="Ornstein-Uhlenbeck process",
                 y_axis_label="Density",
                 x_range=(min(data["bucket"].values), max(data["bucket"].values)),
                 plot_height=400,
                 plot_width=450, toolbar_location='below',
                 toolbar_sticky=False)

    p_v_2 = figure(
                 x_axis_label="Ornstein-Uhlenbeck process",
                 x_range=(min(data["bucket"].values), max(data["bucket"].values)),
                 plot_height=400,
                 plot_width=450, toolbar_location='below',
                 toolbar_sticky=False)

    dps = int(len(data) / steps)
    cds_b = ColumnDataSource(data={
        "top": [],
        "bottom": [],
        "left": [],
        "right": []})

    cds_s = ColumnDataSource(data={
        "top": [],
        "bottom": [],
        "left": [],
        "right": []})

    p_v_1.quad(top='top', bottom=0, left='left', right='right', source=cds_b,
             fill_color='green', line_color="white", alpha=0.4,
               legend_label="Short to Long (Buy)")

    p_v_2.quad(top='top', bottom=0, left='left', right='right', source=cds_s,
             fill_color='red', line_color="white", alpha=0.4,
               legend_label="Long to Short (Sell)")

    def update_plots(attr, old, new):
        lower = (slider_1.value - 1) * dps
        upper = slider_1.value * dps

        data_temp = data.iloc[lower:upper, :]
        data_temp_b = data_temp[data_temp['decision'] == "b"]
        data_temp_s = data_temp[data_temp['decision'] == "s"]

        hist_b, edges_b = np.histogram(data_temp_b['bucket'].values, density=True, bins=7)
        hist_s, edges_s = np.histogram(data_temp_s['bucket'].values, density=True, bins=7)

        new_data_b = {
            "top": hist_b,
            "bottom": np.zeros(len(hist_b)),
            "left": edges_b[:-1],
            "right": edges_b[1:]}

        new_data_s = {
            "top": hist_s,
            "bottom": np.zeros(len(hist_s)),
            "left": edges_s[:-1],
            "right": edges_s[1:]}

        cds_b.data = new_data_b
        cds_s.data = new_data_s

    slider_1 = Slider(start=1, end=steps, step=1, value=1, title='Slider')
    slider_1.on_change('value', update_plots)

    return p_v_1, p_v_2, slider_1

def pnl_dev(scores, bollinger, steps=4, bins=40):


    p_v = figure(title='Dynamic distribution of PnL vs Benchmark (Bollinger Bands)',
                 x_axis_label="PnL",
                 y_axis_label="Density",
                 x_range=(min(scores), max(scores)),
                 plot_height=400,
                 plot_width=900, toolbar_location='below',
                 toolbar_sticky=False)

    hist_boll, edges_boll = np.histogram(bollinger, density=True, bins=bins)
    pdf_boll = gaussian_kde(bollinger)
    x_boll = np.linspace(min(bollinger), max(bollinger), bins)

    cds_boll = ColumnDataSource(data={
        "top": hist_boll,
        "bottom": np.zeros(len(hist_boll)),
        "left": edges_boll[:-1],
        "right": edges_boll[1:],
        "x": x_boll,
        "y": pdf_boll(x_boll)})

    p_v.quad(top='top', bottom=0, left='left', right='right', source=cds_boll,
             fill_color='blue', line_color="white", alpha=0.25)

    p_v.line(x='x', y='y', source=cds_boll, line_color='blue', line_width=2, alpha=1.0,
             legend_label="Benchmark (Bollinger Bands)")

    dps = int(len(scores) / steps)
    cds = ColumnDataSource(data={
        "top": [],
        "bottom": [],
        "left": [],
        "right": [],
        "x": [],
        "y": []})

    p_v.quad(top='top', bottom=0, left='left', right='right', source=cds,
             fill_color='green', line_color="white", alpha=0.4)

    p_v.line(x='x', y='y', source=cds, line_color='green', line_width=2, alpha=1.0,
             legend_label="Dynamic PnL")

    def update_plots(attr, old, new):
        lower = (slider_1.value - 1) * dps
        upper = slider_1.value * dps

        scores_temp = scores[lower:upper]
        hist, edges = np.histogram(scores_temp, density=True, bins=bins)
        pdf = gaussian_kde(scores_temp)
        x = np.linspace(min(scores_temp), max(scores_temp), bins)

        new_data = {
            "top": hist,
            "bottom": np.zeros(len(hist)),
            "left": edges[:-1],
            "right": edges[1:],
            "x": x,
            "y": pdf(x)}

        cds.data = new_data

    slider_1 = Slider(start=1, end=steps, step=1, value=1, title='Slider')
    slider_1.on_change('value', update_plots)

    return p_v, slider_1


def bokeh_page_simple(filename, bollinger, q_learning=False):
    with open(filename, 'rb') as handle:
        output_dict = pickle.load(handle)

    scores = output_dict["scores"]
    # q_tabular = output_dict["q_tabular"]
    data = output_dict["data"]
    data = data[data["decision"]!="n"]


    p, slider = pnl_dev(scores, bollinger, steps=10, bins=20)
    p_1, select = range_plot(data)
    p_v, slider_1 = hist_and_kde(data)
    if q_learning:
        p_v_1, p_v_2, slider_2 = hist_buckets(data)
        return column(p, slider,
                          p_1, select,
                          p_v, slider_1,
                          row(p_v_1, p_v_2), slider_2)
    return column(p, slider,
                          p_1, select,
                          p_v, slider_1,)