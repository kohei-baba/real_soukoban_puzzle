"""Polygon geometry for non-square rigid crates. Meshes are generated locally."""
from pathlib import Path
import math
import struct
import numpy as np


def vertices(cfg):
    shape = cfg.get('shape', 'square')
    if shape == 'triangle':
        radius = cfg.get('shape_radius', .34)
        return np.array([[radius*math.cos(math.pi/2+2*math.pi*i/3),
                          radius*math.sin(math.pi/2+2*math.pi*i/3)] for i in range(3)])
    if shape == 'star':
        points=[]
        for x,y,length,width,yaw in star_arms(cfg):
            c,s=math.cos(yaw),math.sin(yaw)
            for u in (-length/2,length/2):
                for v in (-width/2,width/2):
                    points.append((x+c*u-s*v,y+s*u+c*v))
        return np.asarray(points)
    raise ValueError(f'Unsupported polygon: {shape}')


def star_arms(cfg):
    """Five overlapping rectangular arms; collision and visible geometry match."""
    radius=cfg.get('shape_radius',.35)
    width=cfg.get('shape_arm_width',.13)
    length=radius
    return [(radius/2*math.cos(math.pi/2+2*math.pi*i/5),
             radius/2*math.sin(math.pi/2+2*math.pi*i/5),length,width,
             math.pi/2+2*math.pi*i/5) for i in range(5)]


def support(cfg, yaw, direction):
    """Distance from center to furthest body point in the given world direction."""
    direction = np.asarray(direction, dtype=float)
    if cfg.get('shape', 'square') == 'square':
        a = math.atan2(direction[1], direction[0])
        return cfg['box_size']/2*(abs(math.cos(yaw-a))+abs(math.sin(yaw-a)))
    v = vertices(cfg)
    c, s = math.cos(yaw), math.sin(yaw)
    return float(np.max((v @ np.array([[c, s], [-s, c]])) @ direction))


def angle_error(yaw, cfg):
    n = {'square':4, 'triangle':3, 'star':5}[cfg.get('shape', 'square')]
    period = 2*math.pi/n
    return (yaw+period/2) % period-period/2


def clearance(cfg,yaw,offset):
    """Shortest planar distance from a point to the occupied rigid shape."""
    offset=np.asarray(offset,dtype=float)
    c,s=math.cos(yaw),math.sin(yaw)
    local=np.array([[c,s],[-s,c]])@offset
    shape=cfg.get('shape','square')
    if shape=='square':
        return float(np.linalg.norm(np.maximum(np.abs(local)-cfg['box_size']/2,0)))
    if shape=='star':
        distances=[]
        for x,y,length,width,arm_yaw in star_arms(cfg):
            a=local-np.array([x,y])
            ca,sa=math.cos(arm_yaw),math.sin(arm_yaw)
            arm=np.array([[ca,sa],[-sa,ca]])@a
            distances.append(np.linalg.norm(np.maximum(np.abs(arm)-[length/2,width/2],0)))
        return float(min(distances))
    v=vertices(cfg)
    # Vertices are counter-clockwise. A point inside the triangle has zero clearance.
    if all(np.cross(v[(i+1)%3]-v[i],local-v[i])>=0 for i in range(3)):
        return 0.
    distances=[]
    for i in range(3):
        a,b=v[i],v[(i+1)%3]
        t=np.clip(np.dot(local-a,b-a)/np.dot(b-a,b-a),0.,1.)
        distances.append(np.linalg.norm(local-(a+t*(b-a))))
    return float(min(distances))


def _write_stl(path, vertices_xy, height):
    """A closed prism; polygon is triangulated as a center fan."""
    v = np.asarray(vertices_xy, dtype=float)
    triangles = []
    lo,hi=-height/2,height/2
    for i, p in enumerate(v):
        q = v[(i+1) % len(v)]
        a, b = (float(p[0]),float(p[1])), (float(q[0]),float(q[1]))
        triangles.extend([
            ((0.,0.,lo), (b[0],b[1],lo), (a[0],a[1],lo)),
            ((0.,0.,hi), (a[0],a[1],hi), (b[0],b[1],hi)),
            ((a[0],a[1],lo), (b[0],b[1],lo), (b[0],b[1],hi)),
            ((a[0],a[1],lo), (b[0],b[1],hi), (a[0],a[1],hi)),
        ])
    with Path(path).open('wb') as f:
        f.write(b'RealSokoban polygon prism'.ljust(80,b' '))
        f.write(struct.pack('<I',len(triangles)))
        for a,b,c in triangles:
            va,vb,vc=map(lambda x:np.asarray(x,dtype=float),(a,b,c))
            normal=np.cross(vb-va,vc-va)
            norm=np.linalg.norm(normal)
            if norm: normal/=norm
            f.write(struct.pack('<12fH',*normal,*a,*b,*c,0))


def meshes(directory, cfg, height):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    v=vertices(cfg)
    visual=directory/'shape_visual.stl'
    _write_stl(visual,v,height)
    collision=[]
    if cfg['shape']!='triangle': raise ValueError('Only triangle uses a mesh')
    collision=[visual]
    return visual,collision
