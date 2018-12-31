radar = np.loadtxt(radar_path, delimiter=',')
radar = radar.reshape((4, 60, 60))