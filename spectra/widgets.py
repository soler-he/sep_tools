import ipywidgets as widgets

# a list of available spacecraft:
list_of_sc = ["PSP", "SOHO", "Solar Orbiter", "STEREO-A", "STEREO-B", "Wind"]

stereo_instr = ["HET", "SEPT"]
solo_instr = ["EPT", "HET"]
soho_instr = ["ERNE-HED"]
psp_instr = ["isois-epihi"]
wind_instr = ["3DP"]

sensor_dict = {
    "STEREO-A": stereo_instr,
    "STEREO-B": stereo_instr,
    "Solar Orbiter": solo_instr,
    "SOHO": soho_instr,
    "PSP": psp_instr,
    "Wind": wind_instr
}

view_dict = {
    ("STEREO-A", "SEPT"): ("sun", "asun", "north", "south"),
    ("STEREO-B", "SEPT"): ("sun", "asun", "north", "south"),
    # ("Solar Orbiter", "STEP"): ("Pixel averaged", "Pixel 1", "Pixel 2", "Pixel 3", "Pixel 4", "Pixel 5", "Pixel 6", "Pixel 7", "Pixel 8", "Pixel 9", "Pixel 10",
    #                             "Pixel 11", "Pixel 12", "Pixel 13", "Pixel 14", "Pixel 15"),
    ("Solar Orbiter", "EPT"): ("sun", "asun", "north", "south"),
    ("Solar Orbiter", "HET"): ("sun", "asun", "north", "south"),
    ("PSP", "isois-epihi"): ("A", "B"),
    # ("PSP", "isois-epilo"): ('3', '7'),  # ('0', '1', '2', '3', '4', '5', '6', '7')
    ("Wind", "3DP"): ('omnidirectional', )  # 'sector 0', 'sector 1', 'sector 2', 'sector 3', 'sector 4', 'sector 5', 'sector 6', 'sector 7')
}

species_dict = {
    # ("STEREO-A", "LET"): ("protons", "electrons"),
    ("STEREO-A", "SEPT"): ("ions", "electrons"),
    ("STEREO-A", "HET"): ("protons", "electrons"),
    # ("STEREO-B", "LET"): ("protons", "electrons"),
    ("STEREO-B", "SEPT"): ("ions", "electrons"),
    ("STEREO-B", "HET"): ("protons", "electrons"),
    # ("Solar Orbiter", "STEP"): ("ions",),  # , "electrons"),
    ("Solar Orbiter", "EPT"): ("ions", "electrons"),
    ("Solar Orbiter", "HET"): ("protons", "electrons"),
    ("SOHO", "ERNE-HED"): ("protons",),
    # ("SOHO", "EPHIN"): ("electrons",),
    ("PSP", "isois-epihi"): ("protons", ),  # "electrons"),
    # ("PSP", "isois-epilo"): ("electrons",),
    ("Wind", "3DP"): ("protons", "electrons")
}


# Drop-downs for dynamic particle spectrum:
spacecraft_drop = widgets.Dropdown(options=list_of_sc,
                                   description="Spacecraft:",
                                   disabled=False,
                                   value='Solar Orbiter'
                                   )

sensor_drop = widgets.Dropdown(options=sensor_dict[spacecraft_drop.value],
                               description="Sensor:",
                               disabled=False,
                               )

view_drop = widgets.Dropdown(options=view_dict[(spacecraft_drop.value, sensor_drop.value)],
                             description="Viewing:",
                             disabled=False
                             )

species_drop = widgets.Dropdown(options=species_dict[(spacecraft_drop.value, sensor_drop.value)],
                                description="Species:",
                                disabled=False,
                                )


def update_sensor_options(val):
    """
    this function updates the options in sensor_drop menu
    """
    sensor_drop.options = sensor_dict[spacecraft_drop.value]


def update_view_options(val):
    """
    updates the options and availability of view_drop menu
    """
    try:
        view_drop.disabled = False
        view_drop.options = view_dict[(spacecraft_drop.value, sensor_drop.value)]
        view_drop.value = view_drop.options[0]
    except KeyError:
        view_drop.disabled = True
        view_drop.value = None


def update_species_options(val):
    try:
        species_drop.options = species_dict[(spacecraft_drop.value, sensor_drop.value)]
    except KeyError:
        pass


# makes spacecraft_drop run these functions every time it is accessed by user
spacecraft_drop.observe(update_sensor_options)
spacecraft_drop.observe(update_view_options)
sensor_drop.observe(update_view_options)

# does the same but for sensor menu
spacecraft_drop.observe(update_species_options)
sensor_drop.observe(update_species_options)

# also observe the radio menu
# radio_button.observe(update_radio_options)
