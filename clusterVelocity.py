#!/usr/bin/env python3
# Author: Developed for GeoGateway by Robert Granat and Michael Heflin
# Date: Aug 7, 2019
# Organization: JPL, California Institute of Techology

prolog="""
**PROGRAM**
    clusterVelocities.py

**PURPOSE**
    Calculate cluster membership of GPS stations based on velocities obtained from the routine getVelocities.py.  Return labeled kml and text table files.

    See http://scikit-learn.org/stable/modules/clustering.html for information on individual clustering methods.

**USAGE**
"""
epilog="""
**EXAMPLE**
   clusterVelocity.py -input velocity_table.txt -output labeled_velocity.kml -feature_name "Lat" "Lon" "Delta E" "Delta N" "Delta V" -k 10 --scale -method k-means

**COPYRIGHT**
    | Copyright 2018, by the California Institute of Technology
    | United States Government Sponsorship acknowledged
    | All rights reserved

**AUTHORS**
    | Developed for GeoGateway by Robert Granat and Michael Heflin
    | Jet Propulsion Laboratory
    | California Institute of Technology
    | Pasadena, CA, USA
"""

# Import python modules
import numpy as np
import matplotlib
import matplotlib.cm as cm
import random
import argparse
import math
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

from sklearn import preprocessing

def runCmd(cmd):
    '''run a command'''

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,executable='/bin/bash')
    (out, err) = p.communicate()
    if p.returncode != 0:
        raise UserWarning('failed to run {}\n{}\n'.format(cmd.split()[0],
            err))
    return out

def _getParser():
  parser = argparse.ArgumentParser(description=prolog,epilog=epilog,formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-input', type=str, required=True,
                      help='name of the input velocity file')
  parser.add_argument('-output', type=str, required=True,
                      help='name of the labeled output velocity kml file')
  parser.add_argument('-k', type=int, required=False,
                      help='number of clusters; required for k-means, spectral, agglomerative, gmm, and bgmm; for bgmm represents only an *upper bound* on the number of clusters')
  parser.add_argument('-feature_name', metavar='NAME', type=str, nargs='+', required=True,
                      help='names of features to use in clustering; must be some subset of {"Lon", "Lat", "Delta E", "Delta N", "Delta V", "Sigma E", "Sigma N", "Sigma V"}')
  parser.add_argument('-method', type=str, required=False, default='k-means',
                      help='clustering method; must be one of {k-means, affinity, meanshift, spectral, agglomerative, bdscan, gmm, bgmm} (default: use k-means)')
  parser.add_argument('--scale', action='store_true', default=False, required=False,
                      help='scale features to have zero mean and unit variance (default: no scaling)')
  return parser

def main():
  # Read command line arguments
  parser = _getParser()
  args = parser.parse_args()

  # Read the velocity file and extract features 
  # Assumes Mike Helfin's formatting in getVelocities.py
  df = pd.read_csv(args.input, skiprows=1, names=['Site', 'Lon', 'Lat', 'Delta E', 'Delta N', 'Delta V', 'Sigma E', 'Sigma N', 'Sigma V'], delim_whitespace=True)
  Z = df[args.feature_name]

  # Optional: scale features to have zero mean and unit covariance 
  if args.scale:
    Z = preprocessing.scale(Z)

  # Perform the clustering
  if args.method == 'k-means':
    k = args.k
    kmeans = KMeans(n_clusters = k, init='k-means++', max_iter = 1000, n_init = 10, random_state=1).fit(Z)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
  elif args.method == 'affinity':
    af = AffinityPropagation().fit(Z)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    k = len(cluster_centers_indices)
  elif args.method == 'meanshift':
    ms = MeanShift().fit(Z)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    k = len(labels_unique)
  elif args.method == 'spectral':
    k = args.k
    sc = SpectralClustering(n_clusters = k).fit(Z)
    labels = sc.labels_
  elif args.method == 'agglomerative':
    k = args.k
    ac = AgglomerativeClustering(n_clusters = k).fit(Z)
    labels = ac.labels_
  elif args.method == 'dbscan':
    db = DBSCAN().fit(Z)
    labels = db.labels_
    labels_unique = np.unique(labels)
    k = len(labels_unique)
  elif args.method == 'gmm':
    k = args.k
    gm = GaussianMixture(n_components = k, n_init = 10).fit(Z)
    labels = gm.predict(Z)
  elif args.method == 'bgmm':
    k = args.k
    bgm = BayesianGaussianMixture(n_components = k, n_init = 10).fit(Z)
    labels = bgm.predict(Z)
    labels_unique = np.unique(labels)
    k = len(labels_unique)
  else:
    raise ValueError('Invalid clustering method specified')

  # Rotate labels vector into a column
  labels = np.reshape(labels,(-1,1))

  # Add the labels to the dataframe and then write the results
  df = df.assign(Label=labels)

  # Write outputs; this is a modified version of Mike Heflin's output format

  # Setup color map for markers
  norm = matplotlib.colors.Normalize(vmin=0, vmax=k, clip=False)
  mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)

  # Start kml file
  outFile = open(args.output.partition('.')[0]+'.kml','w')
  print("<?xml version=\"1.0\" encoding=\"UTF-8\"?>",file=outFile)
  print("<kml xmlns=\"http://www.opengis.net/kml/2.2\">",file=outFile)
  print(" <Folder>",file=outFile)

  # Start kml file without vector markers
  outFileNoVec = open(args.output.partition('.')[0]+'_novector.kml','w')
  print("<?xml version=\"1.0\" encoding=\"UTF-8\"?>",file=outFileNoVec)
  print("<kml xmlns=\"http://www.opengis.net/kml/2.2\">",file=outFileNoVec)
  print(" <Folder>",file=outFileNoVec)

  # Start txt file
  txtFile = open(args.output.partition('.')[0]+'.txt','w')
  print("Site          Lon          Lat      Delta E      Delta N      Delta V      Sigma E      Sigma N      Sigma V        Label",file=txtFile)

  # Add markers and vectors
  for i in range(0,df.shape[0]):
    lon = df.iat[i,1]
    lat = df.iat[i,2]
    vlon = df.iat[i,3]
    vlat = df.iat[i,4]
    vrad = df.iat[i,5]
    slon = df.iat[i,6]
    slat = df.iat[i,7]
    srad = df.iat[i,8]
    label = df.iat[i,9]
  
    # Set marker color
    rgba_color = mapper.to_rgba(label, bytes=True)
    markercolor = "#ff{:02x}{:02x}{:02x}".format(rgba_color[0],rgba_color[1],rgba_color[2])

    # Set scale, assuming default from getVelocities
    scale = 320

    # Draw markers for kml output
    print("  <Placemark>",file=outFile)
    print("   <description><![CDATA[",file=outFile)
    print("    <a href=\"http://sideshow.jpl.nasa.gov/post/links/{:s}.html\">".format(df.iat[i,0]),file=outFile)
    print("     <img src=\"http://sideshow.jpl.nasa.gov/post/plots/{:s}.jpg\" width=\"300\" height=\"300\">".format(df.iat[i,0]),file=outFile)
    print("    </a>",file=outFile)
    print("   ]]></description>",file=outFile)
    print("   <Style><IconStyle>",file=outFile)
    print("    <color>{:s}</color>".format(markercolor),file=outFile)
    print("    <scale>0.50</scale>",file=outFile)
    print("    <Icon><href>http://maps.google.com/mapfiles/kml/paddle/wht-blank.png</href></Icon>",file=outFile)
    print("   </IconStyle></Style>",file=outFile)
    print("   <Point>",file=outFile)
    print("    <coordinates>",file=outFile)
    print("     {:f},{:f},0".format(lon,lat),file=outFile)
    print("    </coordinates>",file=outFile)
    print("   </Point>",file=outFile)
    print("  </Placemark>",file=outFile)

    # Draw markers for kml output w/o vectors
    print("  <Placemark>",file=outFileNoVec)
    print("   <description><![CDATA[",file=outFileNoVec)
    print("    <a href=\"http://sideshow.jpl.nasa.gov/post/links/{:s}.html\">".format(df.iat[i,0]),file=outFileNoVec)
    print("     <img src=\"http://sideshow.jpl.nasa.gov/post/plots/{:s}.jpg\" width=\"300\" height=\"300\">".format(df.iat[i,0]),file=outFileNoVec)
    print("    </a>",file=outFileNoVec)
    print("   ]]></description>",file=outFileNoVec)
    print("   <Style><IconStyle>",file=outFileNoVec)
    print("    <color>{:s}</color>".format(markercolor),file=outFileNoVec)
    print("    <scale>0.50</scale>",file=outFileNoVec)
    print("    <Icon><href>http://maps.google.com/mapfiles/kml/paddle/wht-blank.png</href></Icon>",file=outFileNoVec)
    print("   </IconStyle></Style>",file=outFileNoVec)
    print("   <Point>",file=outFileNoVec)
    print("    <coordinates>",file=outFileNoVec)
    print("     {:f},{:f},0".format(lon,lat),file=outFileNoVec)
    print("    </coordinates>",file=outFileNoVec)
    print("   </Point>",file=outFileNoVec)
    print("  </Placemark>",file=outFileNoVec)

    # Draw vectors
    print("  <Placemark>",file=outFile)
    print("   <Style><LineStyle>",file=outFile)
    print("    <color>{:s}</color>".format(markercolor),file=outFile)
    print("    <width>2</width>",file=outFile)
    print("   </LineStyle></Style>",file=outFile)
    print("   <LineString>",file=outFile)
    print("   <coordinates>",file=outFile)
    print("   {:f},{:f},0".format(lon,lat),file=outFile)
    print("   {:f},{:f},0".format(lon+vlon/scale,lat+vlat/scale),file=outFile)
    print("    </coordinates>",file=outFile)
    print("   </LineString>",file=outFile)
    print("  </Placemark>",file=outFile)

    # Draw sigmas
    print("  <Placemark>",file=outFile)
    print("   <Style>",file=outFile)
    print("    <LineStyle>",file=outFile)
    print("     <color>{:s}</color>".format(markercolor),file=outFile)
    print("     <width>2</width>",file=outFile)
    print("    </LineStyle>",file=outFile)
    print("    <PolyStyle>",file=outFile)
    print("     <color>{:s}</color>".format(markercolor),file=outFile)
    print("     <fill>0</fill>",file=outFile)
    print("    </PolyStyle>",file=outFile)
    print("   </Style>",file=outFile)
    print("   <Polygon>",file=outFile)
    print("    <outerBoundaryIs>",file=outFile)
    print("     <LinearRing>",file=outFile)
    print("      <coordinates>",file=outFile)

    theta = 0
    for k in range(0,16):
      angle = k/15*2*math.pi
      elon = slon*math.cos(angle)*math.cos(theta)-slat*math.sin(angle)*math.sin(theta)
      elat = slon*math.cos(angle)*math.sin(theta)+slat*math.sin(angle)*math.cos(theta)
      elon = (elon+vlon)/scale
      elat = (elat+vlat)/scale
      print("      {:f},{:f},0".format(lon+elon,lat+elat),file=outFile)

    print("      </coordinates>",file=outFile)
    print("     </LinearRing>",file=outFile)
    print("    </outerBoundaryIs>",file=outFile)
    print("   </Polygon>",file=outFile)
    print("  </Placemark>",file=outFile)

    # Make table
    print("{:s} {:12f} {:12f} {:12f} {:12f} {:12f} {:12f} {:12f} {:12f} {:12d}".format(df.iat[i,0],lon,lat,vlon,vlat,vrad,slon,slat,srad,label),file=txtFile)

  # Finish kml file
  print(" </Folder>",file=outFile)
  print("</kml>",file=outFile)
  outFile.close()
  txtFile.close()

  # Finish kml file w/o vectors
  print(" </Folder>",file=outFileNoVec)
  print("</kml>",file=outFileNoVec)
  outFile.close()
  txtFile.close()

if __name__ == '__main__':
    main()
