import json
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import argparse


def main(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)["lifetimes"]
    
    
    entries = [
        (datetime.fromisoformat(entry["timestamp"]), entry["lifetime_us"], entry["error_us"])
        for entry in data
    ]
    
    entries.sort(key=lambda x: x[0])
    timestamps, lifetimes, errors = zip(*entries)
    matplotlib.rcParams['timezone'] = 'US/Central'
    
    plt.figure(figsize=(10,5))
    plt.errorbar(timestamps, lifetimes, yerr=errors, fmt='o-', ecolor='black', capsize=1)
    plt.xlabel("Timestamp [CDT]")
    plt.ylabel("Electron lifetime (µs)")
    plt.ylim(0,10e3)
    plt.title("Electron lifetime time series")
    plt.grid(True)
    plt.tight_layout()
    formatter = mdates.DateFormatter("%m/%d/%Y\n%H:%M CDT")
    plt.gca().xaxis.set_major_formatter(formatter)
                                    
    plt.savefig(output_file)   
    
    plt.figure(figsize=(10,5))
    plt.errorbar(timestamps[-50:], lifetimes[-50:], yerr=errors[-50:], fmt='o-', ecolor='black', capsize=1)
    plt.xlabel("Timestamp [CDT]")
    plt.ylabel("Electron lifetime (µs)")
    # plt.ylim(0,5e3)
    plt.title("Electron lifetime time series last 50 points")
    plt.grid(True)
    plt.tight_layout()
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.savefig(output_file+"_last.png")

    #html file 
    fig = px.scatter(
        x=timestamps,
        y=lifetimes,
        error_y=errors,
        title="Electron lifetime time series"
    )
    fig.update_traces(mode="markers")
    
    fig.update_layout(
        xaxis_title="Timestamp [CDT]",
        yaxis_title="Electron lifetime (µs)",
        yaxis_range=[0, 10_000],  # same as 0 to 10e3
        template="plotly_white",
        width=1000,
        height=500
    )
    
    fig.update_xaxes(
        tickformat="%m/%d/%Y<br>%H:%M CDT"  
    )
    fig.write_html(output_file+".html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, default=None, \
                        help="Path to elifetime json file", required=True)
     

    parser.add_argument('--output_file', default=None, \
                        type=str, help='''Output plot file''',
                        required=True)
    
    
    args = parser.parse_args()
    print(args)
    main(**vars(args))
