import numpy as np
from scipy.ndimage import imread
from math import pi, sqrt, sin, cos
import svgwrite as svg
from random import random, choice
from PIL import Image
import argparse 

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
            scl = 1 - float(image[int(x + image.shape[0]/2), int(y + image.shape[1]/2)]) / 256.
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


def write_spiral(points, filename, radius):
    dwg = svg.Drawing(filename)
    s = svg.shapes.Polyline(points)
    s.fill('none')
    dwg.add(s)
    s.stroke('black', width=0.2)
    #dwg.viewbox(minx=0, miny=0, 
    dwg.viewbox(minx=-radius, miny=-radius, 
                width=radius*2, height=radius*2)

    dwg.save()

def roulette(values, total):
    sel = random() * total
    cur = 0
    i = 0
    while cur < sel:
        cur += values[i]
        i += 1
    return i

def walk_line(p1, p2, w,h):
    i = p1[X]
    j = p1[Y]
    dx = p2[X] - p1[X]
    dy = p2[Y] - p1[Y]
    big = max(abs(dy),abs(dy)) + 1.
    di = dx / big
    dj = dy / big
    points = []
    while i >= 0 and i < w and j >= 0 and j < h and abs(i-p1[X]) <= abs(dx) and abs(j-p1[Y]) <= abs(dy):
        points.append((int(i),int(j)))
        i += di
        j += dj
    return points

def random_line(length, w,h):
    x = random()*w
    y = random()*h
    #p1 = choice([(x,0), (x, image.shape[Y]-1)])
    #p2 = choice([(0,y), (image.shape[X]-1, y)])
    p1 = (x,y)
    a = random() * 2*pi
    p2 = (p1[X] + length*cos(a), p1[Y] + length*sin(a))

    return walk_line(p1, p2, w,h)

def random_lines(nlines, length, image):
    imsize = np.product(image.shape)
    threshold = np.sum(image) / imsize
    print(threshold)
    lines = []
    i = 0
    while i < nlines:
        points = random_line(length, *image.shape)
        #points = random_line(length*(1-i/nlines), *image.shape)
        score = sum(image[i,j] for i,j in points) / len(points)
        if score <= threshold:
            lines.append((points[0], points[-1]))
            i += 1
            for x,y in points:
                image[x,y] = 255
            #threshold -= score / imsize
            #threshold += (255.*len(points)) / imsize
            threshold = np.sum(image) / imsize
    return lines

def write_random_lines(lines, filename, w,h, color='black', opacity=1.0):
    dwg = svg.Drawing(filename)
    for line in lines:
        svgline = svg.shapes.Line(*line)
        svgline.fill('none')
        svgline.stroke(color, width=0.25)
        dwg.add(svgline)

    dwg.viewbox(minx=0, miny=0, width=w, height=h)
    dwg.save()

def write_random_lines_rgb(reds, greens, blues, filename, w,h, opacity=1.0):
    dwg = svg.Drawing(filename)
    for red,green,blue in zip(reds, greens, blues):
        svgline = svg.shapes.Line(*red)
        svgline.fill('none')
        svgline.stroke('red', width=0.5)
        dwg.add(svgline)

        svgline = svg.shapes.Line(*green)
        svgline.fill('none')
        svgline.stroke('green', width=0.5, opacity=opacity)
        dwg.add(svgline)

        svgline = svg.shapes.Line(*blue)
        svgline.fill('none')
        svgline.stroke('blue', width=0.5)
        dwg.add(svgline)

    dwg.viewbox(minx=0, miny=0, width=w, height=h)
    dwg.save()

def circle_image(x,y,r,image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if sqrt((i-x)**2 + (j-y)**2) < r:
                image[i,j] = 255
            else:
                image[i,j] = 1
    return image

def normalize_rgb(image):
    r = np.sum(image[:,:,0])
    g = np.sum(image[:,:,1])
    b = np.sum(image[:,:,2])
    rgb = r+g+b
    print(r,g,b, rgb)

    im2 = np.zeros(image.shape, dtype='float64')
    np.copyto(im2, image, casting='unsafe')
    im2[:,:,0] *= (rgb / 3.) / r
    im2[:,:,1] *= (rgb / 3.) / g
    im2[:,:,2] *= (rgb / 3.) / b
    np.copyto(image, im2, casting='unsafe')

    r = np.sum(image[:,:,0])
    g = np.sum(image[:,:,1])
    b = np.sum(image[:,:,2])
    rgb = r+g+b
    print(r,g,b, rgb)
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--spiral', action='store_true', help='Generate a spiral')
    parser.add_argument('-l', '--lines', action='store_true', help='Generate a line drawing')
    parser.add_argument('-n', '--number', type=int, help='Number of points in the spiral (try 500) or lines in the line drawing (try 5000)')
    parser.add_argument('-d', '--distance', type=float, help='The distance between points on the spiral (try 0.5) or the length each line (try 20)')
    parser.add_argument('-o', '--output', help='Filename to write SVG output to')
    parser.add_argument('filename', help='Input PNG filename')

    args = vars(parser.parse_args())

    image = imread(args['filename'])
    #print(im.shape)
    
    if args['spiral']:
        points = np.empty([args['number'],2], dtype='float64')
        spiral(points, args['distance'], 1.0, image)
        write_spiral(points, args['output'], np.min(image.shape)/2)
    elif args['lines']:
        lines = random_lines(args['number'], args['distance'],  image)
        write_random_lines(lines, args['output'], image.shape[X], image.shape[Y])
                
    #normalize_rgb(im)
#    r = np.sum(im[:,:,0])
#    g = np.sum(im[:,:,1])
#    b = np.sum(im[:,:,2])
#    rgb = r+g+b
#    nlines = 5000
#    reds =   random_lines(nlines * r/(rgb/3), 100,  im[:,:,0].copy())
#    greens = random_lines(nlines * g/(rgb/3), 100,  im[:,:,1].copy())
#    blues =  random_lines(nlines * b/(rgb/3), 100,  im[:,:,2].copy())
#    rgb = [len(reds), len(greens), len(blues)]
#    print(rgb, sum(rgb))
#    write_random_lines_rgb(reds, greens, blues, 'test.svg', im.shape[X], im.shape[Y], opacity=0.5)
    
if __name__ == '__main__': main()
