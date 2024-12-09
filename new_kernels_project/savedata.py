from wolframclient.language import wl, wlexpr
from wolframclient.serializers import export
import os
#save data for epsilons
def get_filename():
    script_dir = os.path.dirname(__file__)
    return script_dir+f'/data'

def wolfram_export(data, path):
    #wolfram_expr = wl.List(*[complex(c.real, c.imag) for c in data])
    wolfram_expr = wl.List(data)

    # Specify the output file path for the .mx file
    output_file = path

    # Export the Wolfram Language expression to a .mx file
    export(wolfram_expr, output_file, target_format='wxf')


def save_charge(A, r_and_c):
    filename=get_filename()
    if not os.path.exists(filename):
        os.makedirs(filename)
    print('saved in folder'+filename)
    wolfram_export(r_and_c,f"data/ccharge_A={A}.mx")