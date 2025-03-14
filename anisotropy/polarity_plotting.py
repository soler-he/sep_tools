from matplotlib.colors import Normalize
import numpy as np
# import pandas as pd
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import pyplot as plt
# from sunpy.coordinates import get_horizons_coord, HeliographicStonyhurst
# from seppy.loader.solo import mag_load
# from SOLO_loaders import solo_mag_loader
# from solo_methods import solo_mag_loader
# import datetime as dt


def polarity_gse(Bx, By, r, V=400, delta_angle=10):
    """
    Calculates the magnetic field polarity sector for magnetic field data (Bx_gse, By_gse) 
    from a spacecraft at a (spherical) distance r (AU). 
    Uses the nominal Parker spiral geometry for solar wind speed V (default 400 km/s) for reference.
    delta_angle determines the uncertain angles around 90([90-delta_angle,90+delta_angle]) and 270 
    ([270-delta_angle,270+delta_angle]). RED = INWARD, BLUE = OUTWARD
    """
    au = 1.495978707e8 #astronomical units (km)
    omega = 2*np.pi/(25.38*24*60*60) #solar rotation rate (rad/s)
    #Flip GSE vector to keep the sense of polarity the same as when using RTN
    Bx = -Bx
    By = -By
    #Nominal Parker spiral angle at distance r (AU)
    phi_nominal = np.rad2deg(np.arctan(omega*r*au/V))
    phi_fix = np.zeros(np.size(Bx))
    phi_fix[(Bx>0) & (By>0)] = 360.0
    phi_fix[(Bx<0)] = 180.0
    phi = np.rad2deg(np.arctan(-By/Bx)) + phi_fix
    #Turn the origin to the nominal Parker spiral direction
    phi_relative = phi - phi_nominal
    phi_relative[phi_relative>360] -= 360
    phi_relative[phi_relative<0] += 360
    pol = np.nan*np.zeros(np.size(Bx))
    
    pol[((phi_relative>=0) & (phi_relative<=90.-delta_angle)) | ((phi_relative>=270.+delta_angle) & (phi_relative<=360))] = 1
    pol[(phi_relative>=90.+delta_angle) & (phi_relative<=270.-delta_angle)] = -1
    pol[((phi_relative>=90.-delta_angle) & (phi_relative<=90.+delta_angle)) | ((phi_relative>=270.-delta_angle) & (phi_relative<=270.+delta_angle))] = 0
    #print("RED = INWARD, negative polarity, [90, 270] degrees")
    #print("BLUE = OUTWARD, positive polarity, [270, 360] and [0, 90] degrees")
    return pol, phi_relative


def polarity_rtn(Br,Bt,Bn,r,lat,V=400,delta_angle=10):
    """
    Calculates the magnetic field polarity sector for magnetic field data (Br, Bt, Bn) 
    from a spacecraft at a (spherical) distance r (AU) and heliographic latitude lat (deg). 
    Uses the nominal Parker spiral geometry for solar wind speed V (default 400 km/s) for reference.
    delta_angle determines the uncertain angles around 90([90-delta_angle,90+delta_angle]) and 270 
    ([270-delta_angle,270+delta_angle]). RED = INWARD, BLUE = OUTWARD
    """
    au = 1.495978707e8 #astronomical units (km)
    omega = 2*np.pi/(25.38*24*60*60) #solar rotation rate (rad/s)
    #Nominal Parker spiral angle at distance r (AU)
    phi_nominal = np.rad2deg(np.arctan(omega*r*au/V))
    #Calculating By and Bx from the data (heliographical coordinates, where meridian centered at sc)
    Bx = Br*np.cos(np.deg2rad(lat)) - Bn*np.sin(np.deg2rad(lat))
    By = Bt
    phi_fix = np.zeros(np.size(Bx))
    phi_fix[(Bx>0) & (By>0)] = 360.0
    phi_fix[(Bx<0)] = 180.0
    phi = np.rad2deg(np.arctan(-By/Bx)) + phi_fix
    #Turn the origin to the nominal Parker spiral direction
    phi_relative = phi - phi_nominal
    phi_relative[phi_relative>360] -= 360
    phi_relative[phi_relative<0] += 360
    pol = np.nan*np.zeros(np.size(Br))
    
    pol[((phi_relative>=0) & (phi_relative<=90.-delta_angle)) | ((phi_relative>=270.+delta_angle) & (phi_relative<=360))] = 1
    pol[(phi_relative>=90.+delta_angle) & (phi_relative<=270.-delta_angle)] = -1
    pol[((phi_relative>=90.-delta_angle) & (phi_relative<=90.+delta_angle)) | ((phi_relative>=270.-delta_angle) & (phi_relative<=270.+delta_angle))] = 0
    print("RED = INWARD, negative polarity, [90, 270] degrees")
    print("BLUE = OUTWARD, positive polarity, [270, 360] and [0, 90] degrees")
    return pol, phi_relative


def polarity_colorwheel():
    # Generate a figure with a polar projection
    fg = plt.figure(figsize=(1,1))
    ax = fg.add_axes([0.1,0.1,0.8,0.8], projection='polar')

    n = 100  #the number of secants for the mesh
    norm = Normalize(0, np.pi) 
    t = np.linspace(0,np.pi,n)   #theta values
    r = np.linspace(.6,1,2)        #radius values change 0.6 to 0 for full circle
    rg, tg = np.meshgrid(r,t)      #create a r,theta meshgrid
    c = tg                         #define color values as theta value
    im = ax.pcolormesh(t, r, c.T,norm=norm,cmap="bwr")  #plot the colormesh on axis with colormap
    t = np.linspace(np.pi,2*np.pi,n)   #theta values
    r = np.linspace(.6,1,2)        #radius values change 0.6 to 0 for full circle
    rg, tg = np.meshgrid(r,t)      #create a r,theta meshgrid
    c = 2*np.pi-tg                         #define color values as theta value
    im = ax.pcolormesh(t, r, c.T,norm=norm,cmap="bwr")  #plot the colormesh on axis with colormap
    ax.set_yticklabels([])                   #turn of radial tick labels (yticks)
    ax.tick_params(pad=0,labelsize=8)      #cosmetic changes to tick labels
    ax.spines['polar'].set_visible(False)    #turn off the axis spine.
    ax.grid(False)


def polarity_panel(ax,datetimes,phi_relative,bbox_to_anchor=(0.,0.22,1,1.1),height="10%",x_text1=1.01,x_text2=1.04,y_text=0.1,legend_font=8.5):
    pol_ax = inset_axes(ax, height=height, width="100%", loc=9, bbox_to_anchor=bbox_to_anchor, bbox_transform=ax.transAxes)
    pol_ax.get_xaxis().set_visible(False)
    pol_ax.get_yaxis().set_visible(False)
    pol_ax.set_ylim(0,1)
    pol_arr = np.zeros(len(phi_relative))+1
    timestamp = datetimes[2] - datetimes[1]
    norm = Normalize(vmin=0, vmax=180, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
    pol_ax.bar(datetimes[(phi_relative>=0) & (phi_relative<180)],pol_arr[(phi_relative>=0) & (phi_relative<180)],color=mapper.to_rgba(phi_relative[(phi_relative>=0) & (phi_relative<180)]),width=timestamp)
    pol_ax.bar(datetimes[(phi_relative>=180) & (phi_relative<360)],pol_arr[(phi_relative>=180) & (phi_relative<360)],color=mapper.to_rgba(np.abs(360-phi_relative[(phi_relative>=180) & (phi_relative<360)])),width=timestamp)
    pol_ax.text(x_text1,y_text,"in",color="red",transform=pol_ax.transAxes,fontsize=legend_font-0.5)
    pol_ax.text(x_text2,y_text,"out",color="blue",transform=pol_ax.transAxes,fontsize=legend_font-0.5)
    pol_ax.set_xlim(ax.get_xlim())
    return pol_ax
