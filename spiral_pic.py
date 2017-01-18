import numpy as np
from scipy.ndimage import imread
from math import pi
import svgwrite as svg
from random import random

X = 0
Y = 1
def in_image(x,y, image):
    return x + image.shape[0]/2 < image.shape[0] and \
           x + image.shape[0]/2 >= 0 and \
           y + image.shape[1]/2 < image.shape[1] and \
           y + image.shape[1]/2 >= 0

def spiral(points, step_along_spiral, step_out_per_rot, image):
    dr = step_out_per_rot / (2*pi)
    r = step_along_spiral
    a = r / dr
    x,y = 0,0
    npoints = 0
    while r < np.min(image.shape)/2:
        if points.shape[0] <= npoints:
            points.resize((npoints + 100000, points.shape[1]), refcheck=False)
            print('Resize to %s points for radius %s/%s'%(points.shape[0], r, np.min(image.shape)/2))

        a += step_along_spiral / r
        r = dr * a  

        x = r * np.cos(a) 
        y = r * np.sin(a) 

        if in_image(x,y, image):
            scl = 1 - float(image[x + image.shape[0]/2,y + image.shape[1]/2]) / 256.
            #points[i][X] = (r + scl * np.sin(i * scl * pi/2)) * np.cos(a) 
            #points[i][Y] = (r + scl * np.sin(i * scl * pi/2)) * np.sin(a) 
            points[npoints][X] = (r + scl * step_out_per_rot * (0.5 - random())) * np.cos(a) 
            points[npoints][Y] = (r + scl * step_out_per_rot * (0.5 - random())) * np.sin(a) 
            #points[i][X] = (r + scl * 1.5 * (i%2)) * np.cos(a) 
            #points[i][Y] = (r + scl * 1.5 * (i%2)) * np.sin(a) 
        else:
            points[npoints][X] = x
            points[npoints][Y] = y

        npoints += 1
    print('%s points'%npoints)
    points.resize((npoints, points.shape[1]), refcheck=False)



def main():
    npoints = 100000
    points = np.empty([npoints,2], dtype='float64')

    im = imread('greyella.png')
    print("Read %s,%s pixels e.g. %s"%(im.shape[0], im.shape[1], im[256,256]))
    #im = imread('gradient.png')
    spiral(points, 0.5, 1.0, im)
    
    dwg = svg.Drawing('test.svg')
    s = svg.shapes.Polyline(points)
    s.fill('none')
    dwg.add(s)
    s.stroke('black', width=0.2)
    #dwg.viewbox(minx=0, miny=0, 
    r = np.min(im.shape)/2
    dwg.viewbox(minx=-r, miny=-r, 
                width=r*2, height=r*2)

    dwg.save()
if __name__ == '__main__': main()
