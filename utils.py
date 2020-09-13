#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np
import matplotlib.pyplot as plt
import folium
import pandas as pd
import math
import numpy as np

def set_pyplot_options(figsize=30, font="serif"):
    plt.rcParams['figure.figsize'] = [figsize, figsize]
    plt.rcParams["font.family"] = font
    #print(plt.rcParams)

def geodetic_to_ecef(lat, lon, h):
    # (lat, lon) in WSG-84 degrees
    # h in meters
    lamb = np.radians(lat)
    phi = np.radians(lon)
    s = np.sin(lamb)
    N = a / np.sqrt(1 - e_sq * s * s)

    sin_lambda = np.sin(lamb)
    cos_lambda = np.cos(lamb)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    x = (h + N) * cos_lambda * cos_phi
    y = (h + N) * cos_lambda * sin_phi
    z = (h + (1 - e_sq) * N) * sin_lambda

    return [x, y, z]

def ecef_to_enu(x, y, z, lat0, lon0, h0):
    lamb = np.radians(lat0)
    phi = np.radians(lon0)
    s = np.sin(lamb)
    N = a / np.sqrt(1 - e_sq * s * s)

    sin_lambda = np.sin(lamb)
    cos_lambda = np.cos(lamb)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda

    xd = x - x0
    yd = y - y0
    zd = z - z0

    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
    zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

    return [xEast, yNorth, zUp]

# +
def geodetic_to_enu(lat, lon, h, lat_ref, lon_ref, h_ref):
    [x, y, z] = geodetic_to_ecef(lat, lon, h)
    
    return ecef_to_enu(x, y, z, lat_ref, lon_ref, h_ref)

def NMEAtoDeg(lat):
    lat=np.array(lat)
    # Convert latitude, longitude to degree
    deg_int = np.floor(lat*0.01)
    return deg_int + (lat-deg_int*100)/60


# -

a = 6378137.0
b = 6356752.314245
f = (a - b) / a
e_sq = f * (2-f)
lat0 = 36.0124
lon0 = 129.3186
h0 = 80



dx, dy, _ = geodetic_to_enu(lat0+0.0001,lon0+0.0001,0, lat0, lon0, h0)
background_ratio = dy/dx

def show_gps(df, idx_start=0, idx_end=None, enu_format=True, annotate=False):
    
    time = df[['time']][idx_start:idx_end].to_numpy()
    lat  = df[['latitude']][idx_start:idx_end].to_numpy()
    lon  = df[['longitude']][idx_start:idx_end].to_numpy()
    enu_x = df[['enu_x']][idx_start:idx_end].to_numpy()
    enu_y = df[['enu_y']][idx_start:idx_end].to_numpy()

    pos_range = [129.3186, 129.3202, 36.0124, 36.0140] # POSTECH Field
    
    if(enu_format):
        lat = enu_y
        lon = enu_x
    else:
        #plt.axis(pos_range)
        img = plt.imread("background2.png")
        plt.imshow(img, extent=pos_range)

    if(annotate):
        for idx, xy in enumerate(zip(lon,lat)):
            if(idx % 10 == 0):
                plt.annotate('{}'.format(int(df.index[idx])), xy=xy, textcoords='data',fontsize=30)
        #       plt.annotate('{},{:.3f},{:.3f}'.format(idx, xy[0], xy[1]), xy=xy, textcoords='data')

    plt.plot(lon,lat,'bo', markersize=10, label='GPS Path')
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.axes().set_aspect(background_ratio)
    plt.grid(b=True)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.legend()
    plt.show()

    
def show_imu(df, idx_start=1, idx_end=None):
        
    time = df[['time']][idx_start:idx_end].to_numpy()
    acx  = df[['accX']][idx_start:idx_end].to_numpy()
    acy  = df[['accY']][idx_start:idx_end].to_numpy()
    acz  = df[['accZ']][idx_start:idx_end].to_numpy()
    gyx  = df[['gyrX']][idx_start:idx_end].to_numpy()
    gyy  = df[['gyrY']][idx_start:idx_end].to_numpy()
    gyz  = df[['gyrZ']][idx_start:idx_end].to_numpy()
    atr  = df[['eulR']][idx_start:idx_end].to_numpy()
    atp  = df[['eulP']][idx_start:idx_end].to_numpy()
    aty  = df[['eulY']][idx_start:idx_end].to_numpy()
    
    plt.figure(10)
    plt.subplot(3,1,1)
    plt.plot(time, acx,'r-', label='linAccX')
    plt.plot(time, acy,'b-', label='linAccY')
    plt.plot(time, acz,'g-', label='linAccZ')
    plt.ticklabel_format(style='plain', useOffset=False, axis='both')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(time, gyx,'r-', label='gyroX')
    plt.plot(time, gyy,'b-', label='gyroY')
    plt.plot(time, gyz,'g-', label='gyroZ')
    plt.ticklabel_format(style='plain', useOffset=False, axis='both')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(time, atr,'r-', label='Role')
    plt.plot(time, atp,'b-', label='Pitch')
    plt.plot(time, aty,'g-', label='Yaw')
    plt.legend()
    plt.xlabel('time (10ms count)')
    plt.ticklabel_format(style='plain', useOffset=False, axis='both')

    plt.show()
    

def show_gps_folium(df):
    mapboxAccessToken = 'pk.eyJ1IjoiaHl1bnN1bmdraW0iLCJhIjoiY2tiMWw0eTBuMDFvMzJwcGhqZjQwOWo3eCJ9.6z1Urc6LYwV0Hqo7grj0ag'
    mapboxTilesetId = 'mapbox.satellite'
    
    center = [lat0, lon0]
    df = df.dropna(subset=['longitude'])
    lines = df[['latitude', 'longitude']].values.tolist()
    m = folium.Map(
        location=center,
        color = '#F52C2C',
#        tiles='Stamen Toner',
        tiles='https://api.tiles.mapbox.com/v4/' + mapboxTilesetId + '/{z}/{x}/{y}.png?access_token=' + mapboxAccessToken,
        attr='mapbox.com',
        zoom_start=50
    )
    folium.PolyLine(
        locations = lines
    ).add_to(m)
    
    return m


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


# %%




