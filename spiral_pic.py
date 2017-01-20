import numpy as np
from scipy.ndimage import imread
from math import pi, sqrt, sin, cos
import svgwrite as svg
from random import random, choice
from PIL import Image

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


def write_spiral(points, filename):
    dwg = svg.Drawing(filename)
    s = svg.shapes.Polyline(points)
    s.fill('none')
    dwg.add(s)
    s.stroke('black', width=0.2)
    #dwg.viewbox(minx=0, miny=0, 
    r = np.min(im.shape)/2
    dwg.viewbox(minx=-r, miny=-r, 
                width=r*2, height=r*2)

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
        points = random_line(length*(1-i/nlines), *image.shape)
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

def write_random_lines(lines, filename, w,h):
    dwg = svg.Drawing(filename)
    for line in lines:
        svgline = svg.shapes.Line(*line)
        svgline.fill('none')
        svgline.stroke('black', width=0.5)
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

def main():
    npoints = 500
    points = np.empty([npoints,2], dtype='float64')

    #im = imread('greyella_hi.png')
    #im = imread('greyella.png')
    im = imread('greyella_square.png')
    #im = imread('gradient.png')
    #im = circle_image(20,20, 20, im)
    #im = np.zeros((200,200))

    print("Read %s,%s pixels"%(*im.shape)

    #spiral(points, 0.5, 1.0, im)
    #write_spiral(points, 'test.svg')

    lines = random_lines(10000, 100,  im)
    write_random_lines(lines, 'test.svg', *im.shape)
    #im = Image.frombuffer('L', im.shape, im) # L = grey
    #im.save('test.png')

    
if __name__ == '__main__': main()
