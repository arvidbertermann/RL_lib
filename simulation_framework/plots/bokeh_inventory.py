from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.models import RangeTool
from bokeh.models import BoxSelectTool
from bokeh.models import LassoSelectTool
from bokeh.models import WheelZoomTool
from bokeh.models import BoxZoomTool
from bokeh.models import ResetTool
from bokeh.models import ColumnDataSource
from simulation_framework.plots.bokeh import *
import numpy as np
from scipy.stats.kde import gaussian_kde
from bokeh.models import Slider
import pickle
from bokeh.layouts import row, column
from bokeh.transform import jitter

def range_plot_inv_sell(data):
    data_ss = data[data['decision'] == -2]
    data_s = data[data['decision'] == -1]
    data_b = data[data['decision'] == 1]
    data_bb = data[data['decision'] == 2]

    cds_ss = make_cds(data_ss)
    cds_s = make_cds(data_s)
    cds_b = make_cds(data_b)
    cds_bb = make_cds(data_bb)

    end = 10_000
    p_sell = figure(title='Investment decisions',
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
                     ('Episode', '@episode'),
                     ('Exposure', '@exposure')])], toolbar_location='below',
                 toolbar_sticky=False)
    p_sell.add_tools(LassoSelectTool())
    p_sell.add_tools(WheelZoomTool())
    p_sell.add_tools(BoxZoomTool())
    p_sell.add_tools(ResetTool())
    p_sell.add_tools(BoxSelectTool())
    p_sell.circle(x='time', y='price', alpha=0.6, source=cds_s,
               color='orange', legend_label="Single Sell")
    p_sell.circle(x='time', y='price', alpha=0.6, source=cds_ss,
               color='red', legend_label="Double Sell")

    select = figure(title="Range Determination",
                    x_axis_type="linear",
                    plot_height=130, plot_width=900,
                    y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")

    range_tool = RangeTool(x_range=p_sell.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select.circle(x='time', y='price', alpha=0.1, source=cds_s,
                  color='orange')
    select.circle(x='time', y='price', alpha=0.1, source=cds_ss,
                  color='red')

    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool
    p_sell.legend.location = 'center_left'
    return p_sell, select

def range_plot_inv_buy(data):
    data_ss = data[data['decision'] == -2]
    data_s = data[data['decision'] == -1]
    data_b = data[data['decision'] == 1]
    data_bb = data[data['decision'] == 2]

    cds_ss = make_cds(data_ss)
    cds_s = make_cds(data_s)
    cds_b = make_cds(data_b)
    cds_bb = make_cds(data_bb)

    end = 10_000
    p_buy = figure(title='Investment decisions',
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
                     ('Episode', '@episode'),
                     ('Exposure', '@exposure')])], toolbar_location='below',
                 toolbar_sticky=False)
    p_buy.add_tools(LassoSelectTool())
    p_buy.add_tools(WheelZoomTool())
    p_buy.add_tools(BoxZoomTool())
    p_buy.add_tools(ResetTool())
    p_buy.add_tools(BoxSelectTool())
    p_buy.circle(x='time', y='price', alpha=0.6, source=cds_b,
               color='yellow', legend_label="Single Buy")
    p_buy.circle(x='time', y='price', alpha=0.6, source=cds_bb,
               color='green', legend_label="Double Buy")

    select = figure(title="Range Determination",
                    x_axis_type="linear",
                    plot_height=130, plot_width=900,
                    y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")

    range_tool = RangeTool(x_range=p_buy.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select.circle(x='time', y='price', alpha=0.1, source=cds_b,
                  color='yellow')
    select.circle(x='time', y='price', alpha=0.1, source=cds_bb,
                  color='green')

    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool
    p_buy.legend.location = 'center_left'
    return p_buy, select

def make_cds(data):
    var_list = data.columns
    cds = ColumnDataSource(data={
        var_list[0]: data[var_list[0]],
        var_list[1]: data[var_list[1]],
        var_list[2]: data[var_list[2]],
        var_list[3]: data[var_list[3]],
        var_list[4]: data[var_list[4]],
        var_list[5]: data[var_list[5]],
        var_list[6]: data[var_list[6]],
        var_list[7]: data[var_list[7]],
        var_list[8]: data[var_list[8]],
        var_list[9]: data[var_list[9]]})
    return cds



def hist_and_kde_inventory(data, steps=10, bins=100):
    p_v = figure(title='Dynamic distribution of decisions',
                 x_axis_label="Ornstein-Uhlenbeck process",
                 y_axis_label="Density",
                 x_range=(min(data["price"].values), max(data["price"].values)),
                 plot_height=400,
                 plot_width=900, toolbar_location='below',
                 toolbar_sticky=False)

    dps = int(len(data) / steps)
    cds_ss = empty_cds()
    cds_s = empty_cds()
    cds_b = empty_cds()
    cds_bb = empty_cds()


    p_v.quad(top='top', bottom=0, left='left', right='right', source=cds_ss,
             fill_color='red', line_color="white", alpha=0.4)
    p_v.quad(top='top', bottom=0, left='left', right='right', source=cds_s,
             fill_color='orange', line_color="white", alpha=0.4)
    p_v.quad(top='top', bottom=0, left='left', right='right', source=cds_b,
             fill_color='yellow', line_color="white", alpha=0.4)
    p_v.quad(top='top', bottom=0, left='left', right='right', source=cds_bb,
             fill_color='green', line_color="white", alpha=0.4)

    p_v.line(x='x', y='y', source=cds_ss, line_color='red', line_width=2, alpha=1.0,
             legend_label="Double Sell")
    p_v.line(x='x', y='y', source=cds_s, line_color='orange', line_width=2, alpha=1.0,
             legend_label="Single Sell")
    p_v.line(x='x', y='y', source=cds_b, line_color='yellow', line_width=2, alpha=1.0,
             legend_label="Single Buy")
    p_v.line(x='x', y='y', source=cds_bb, line_color='green', line_width=2, alpha=1.0,
             legend_label="Double Buy")

    def update_plots(attr, old, new):
        lower = (slider_1.value - 1) * dps
        upper = slider_1.value * dps

        data_temp = data.iloc[lower:upper, :]
        data_temp_ss = data_temp[data_temp['decision'] == -2]
        data_temp_s = data_temp[data_temp['decision'] == -1]
        data_temp_b = data_temp[data_temp['decision'] == 1]
        data_temp_bb = data_temp[data_temp['decision'] == 2]

        cds_ss.data = new_data_func(data_temp_ss, 'price', bins)
        cds_s.data = new_data_func(data_temp_s, 'price',bins)
        cds_b.data = new_data_func(data_temp_b, 'price', bins)
        cds_bb.data = new_data_func(data_temp_bb, 'price', bins)

    slider_1 = Slider(start=1, end=steps, step=1, value=1, title='Slider')
    slider_1.on_change('value', update_plots)

    return p_v, slider_1


def empty_cds():
    cds = ColumnDataSource(data={
        "top": [],
        "bottom": [],
        "left": [],
        "right": [],
        "x": [],
        "y": []})
    return cds

def new_data_func(data, column, bins):
    hist, edges = np.histogram(data[column].values, density=True, bins=bins)
    pdf = gaussian_kde(data[column].values)
    x = np.linspace(min(data[column].values), max(data[column].values), bins)

    new_data = {
        "top": hist,
        "bottom": np.zeros(len(hist)),
        "left": edges[:-1],
        "right": edges[1:],
        "x": x,
        "y": pdf(x)}
    return new_data

def hist_buckets_inventory(data, steps=10):
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

    p_v_3 = figure(
        x_axis_label="Frequency of decisions per bucket",
        x_range=(min(data["bucket"].values), max(data["bucket"].values)),
        plot_height=400,
        plot_width=450, toolbar_location='below',
        toolbar_sticky=False)

    p_v_4 = figure(
        x_axis_label="Ornstein-Uhlenbeck process",
        x_range=(min(data["bucket"].values), max(data["bucket"].values)),
        plot_height=400,
        plot_width=450, toolbar_location='below',
        toolbar_sticky=False)


    dps = int(len(data) / steps)
    cds_ss = empty_cds()
    cds_s = empty_cds()
    cds_b = empty_cds()
    cds_bb = empty_cds()

    p_v_1.quad(top='top', bottom=0, left='left', right='right', source=cds_ss,
             fill_color='red', line_color="white", alpha=0.4,
               legend_label="Double Sell")
    p_v_2.quad(top='top', bottom=0, left='left', right='right', source=cds_s,
             fill_color='orange', line_color="white", alpha=0.4,
               legend_label="Single Sell")
    p_v_3.quad(top='top', bottom=0, left='left', right='right', source=cds_b,
               fill_color='yellow', line_color="white", alpha=0.4,
               legend_label="Single Buy")
    p_v_4.quad(top='top', bottom=0, left='left', right='right', source=cds_bb,
               fill_color='green', line_color="white", alpha=0.4,
               legend_label="Double Buy")

    def update_plots(attr, old, new):
        lower = (slider_1.value - 1) * dps
        upper = slider_1.value * dps

        data_temp = data.iloc[lower:upper, :]
        data_temp_ss = data_temp[data_temp['decision'] == -2]
        data_temp_s = data_temp[data_temp['decision'] == -1]
        data_temp_b = data_temp[data_temp['decision'] == 1]
        data_temp_bb = data_temp[data_temp['decision'] == 2]

        cds_ss.data = new_data_func(data_temp_ss, 'bucket', 7)
        cds_s.data = new_data_func(data_temp_s, 'bucket', 7)
        cds_b.data = new_data_func(data_temp_b, 'bucket', 7)
        cds_bb.data = new_data_func(data_temp_bb, 'bucket', 7)


    slider_1 = Slider(start=1, end=steps, step=1, value=1, title='Slider')
    slider_1.on_change('value', update_plots)

    return p_v_1, p_v_2, p_v_3, p_v_4, slider_1

def exposure_uo_scatter(data, steps=10):
    dps = int(len(data) / steps)
    var_list = data.columns
    cds = ColumnDataSource(data={
        var_list[0]: [],
        var_list[1]: [],
        var_list[2]: [],
        var_list[3]: [],
        var_list[4]: [],
        var_list[5]: [],
        var_list[6]: [],
        var_list[7]: [],
        var_list[8]: [],
        var_list[9]: []})
    p = figure(title='Exposure v Ornstein-Uhlenbeck process',
                    x_axis_location="above",
                    x_axis_label='Exposure',
                    y_axis_label="Ornstein-Uhlenbeck process",
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
                        ('Episode', '@episode'),
                        ('Exposure', '@exposure')])], toolbar_location='below',
                    toolbar_sticky=False)
    p.add_tools(LassoSelectTool())
    p.add_tools(WheelZoomTool())
    p.add_tools(BoxZoomTool())
    p.add_tools(ResetTool())
    p.add_tools(BoxSelectTool())
    p.circle(x=jitter('exposure', 0.6), y='price', alpha=0.05, source=cds,
                  color='blue', legend_label="Data points")

    def update_plots(attr, old, new):
        lower = (slider.value - 1) * dps
        upper = slider.value * dps

        data_temp = data.iloc[lower:upper, :]
        var_list = data_temp.columns
        cds.data = {var_list[0]: data_temp[var_list[0]],
            var_list[1]: data_temp[var_list[1]],
            var_list[2]: data_temp[var_list[2]],
            var_list[3]: data_temp[var_list[3]],
            var_list[4]: data_temp[var_list[4]],
            var_list[5]: data_temp[var_list[5]],
            var_list[6]: data_temp[var_list[6]],
            var_list[7]: data_temp[var_list[7]],
            var_list[8]: data_temp[var_list[8]],
            var_list[9]: data_temp[var_list[9]]}


    slider = Slider(start=1, end=steps, step=1, value=1, title='Slider')
    slider.on_change('value', update_plots)

    return p, slider

def exposure_hist_and_kde(data, steps=10, bins=30):
    p_v = figure(title='Dynamic distribution of exposure',
                 x_axis_label="Exposure",
                 y_axis_label="Density",
                 x_range=(min(data["exposure"].values), max(data["exposure"].values)),
                 plot_height=400,
                 plot_width=900, toolbar_location='below',
                 toolbar_sticky=False)

    dps = int(len(data) / steps)
    cds = empty_cds()

    p_v.quad(top='top', bottom=0, left='left', right='right', source=cds,
             fill_color='blue', line_color="white", alpha=0.4)

    p_v.line(x='x', y='y', source=cds, line_color='blue', line_width=2, alpha=1.0,
             legend_label="Exposure")

    def update_plots(attr, old, new):
        lower = (slider.value - 1) * dps
        upper = slider.value * dps

        data_temp = data.iloc[lower:upper, :]
        cds.data = new_data_func(data_temp, 'exposure', bins)


    slider = Slider(start=1, end=steps, step=1, value=1, title='Slider')
    slider.on_change('value', update_plots)

    return p_v, slider


def bokeh_page_inventory(filename, bollinger, q_learning=False):
    with open(filename, 'rb') as handle:
        output_dict = pickle.load(handle)

    scores = output_dict["scores"]
    # q_tabular = output_dict["q_tabular"]
    data = output_dict["data"]

    p, slider = pnl_dev(scores, bollinger, steps=20, bins=20)
    p_sell, select_sell = range_plot_inv_sell(data)
    p_buy, select_buy = range_plot_inv_buy(data)
    p_v, slider_1 =hist_and_kde_inventory(data, steps=20)


    if q_learning:
        p_v_1, p_v_2, p_v_3, p_v_4, slider_2 = hist_buckets_inventory(data, steps=20)
        p_exp, slider_exp = exposure_uo_scatter(data, steps=50)
        return column(p, slider,
                      p_sell, select_sell,
                      p_buy, select_buy,
                      p_v, slider_1,
                      row(p_v_1, p_v_2),
                      slider_2,
                      row(p_v_3, p_v_4),
                      p_exp, slider_exp)

    p_exp, slider_exp = exposure_hist_and_kde(data, steps=10, bins=30)
    return column(p, slider,
                  p_sell, select_sell,
                  p_buy, select_buy,
                  p_v, slider_1,
                  p_exp, slider_exp)
