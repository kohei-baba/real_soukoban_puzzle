"""Self-contained SDF: no remote models, force-driven cylinder, rigid crate."""
from pathlib import Path
import math
from .shapes import meshes,star_arms

def material(color):
    return f'<material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>'

def box_geom(x,y,z): return f'<geometry><box><size>{x} {y} {z}</size></box></geometry>'

def static_box(name,xy,size,color,collision=True,z=None):
    geom=box_geom(*size); z=size[2]/2 if z is None else z
    coll=f'<collision name="collision">{geom}</collision>' if collision else ''
    return f'''<model name="{name}"><static>true</static>
      <pose>{xy[0]} {xy[1]} {z} 0 0 0</pose><link name="body">{coll}
      <visual name="visual">{geom}{material(color)}</visual></link></model>'''

def write_world(path,board,cfg,camera=False):
    cell=board.cell; size=cfg['box_size']; radius=cfg['robot_radius']
    bh,rh=.22,.16; bx,by=board.xy(board.box); rx,ry=board.xy(board.robot)
    w,h=board.cols*cell,board.rows*cell
    pieces=[f'''<?xml version="1.0"?><sdf version="1.9"><world name="real_sokoban">
    <gravity>0 0 -9.81</gravity>
    <physics name="physics" type="ignored"><max_step_size>{cfg['physics_dt']}</max_step_size>
      <real_time_factor>{.5 if camera else 0}</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="RealSokoban" name="rs::RealSokoban">
      <robot_z>{rh/2+.002}</robot_z><box_z>{bh/2+.002}</box_z></plugin>
    <scene><ambient>0.65 0.65 0.65 1</ambient><background>0.06 0.08 0.12 1</background>
      <shadows>false</shadows></scene>
    <light name="sun" type="directional"><pose>0 0 8 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse><specular>0.1 0.1 0.1 1</specular>
      <direction>-0.3 0.2 -1</direction></light>''']
    pieces.append(static_box('floor',(0,0),(w+.5,h+.5,.1),'0.15 0.18 0.23 1',z=-.05))
    for r in range(board.rows):
        for c in range(board.cols):
            if (r,c) in board.walls:
                pieces.append(static_box(f'wall_{r}_{c}',board.xy((r,c)),
                    (cell,cell,.28),'0.35 0.43 0.52 1'))
            else:
                color='0.82 0.85 0.88 1' if (r+c)%2 else '0.7 0.75 0.8 1'
                pieces.append(static_box(f'tile_{r}_{c}',board.xy((r,c)),
                    (cell-.012,cell-.012,.002),color,False,z=.002))
    pieces.append(static_box('goal',board.xy(board.goal),(.7*cell,.7*cell,.003),
                             '0.12 0.8 0.42 1',False,z=.005))
    mass=1.; ix=mass*(size*size+bh*bh)/12; iz=mass*size*size/6
    geom=box_geom(size,size,bh)
    if cfg.get('shape','square')=='star':
        iz=mass*cfg.get('shape_radius',.35)**2*.27
        ix=iy=(iz+mass*bh*bh/6)/2
        arms=[]
        for i,(x,y,length,width,yaw) in enumerate(star_arms(cfg)):
            geom=box_geom(length,width,bh)
            pose=f'<pose>{x} {y} 0 0 0 {yaw}</pose>'
            arms.append(f'''<collision name="collision_{i}">{pose}{geom}
              <surface><friction><ode><mu>0.45</mu><mu2>0.45</mu2></ode></friction></surface>
              </collision><visual name="arm_{i}">{pose}{geom}{material('0.95 0.59 0.12 1')}</visual>''')
        pieces.append(f'''<model name="crate"><pose>{bx} {by} {bh/2+.002} 0 0 0</pose>
          <link name="body"><inertial><mass>{mass}</mass><inertia>
          <ixx>{ix}</ixx><iyy>{iy}</iyy><izz>{iz}</izz></inertia></inertial>
          {''.join(arms)}<visual name="angle_marker"><pose>0 0 {bh/2+.003} 0 0 0</pose>
          {box_geom(.025,.25,.003)}{material('0.3 0.12 0.08 1')}</visual>
          </link></model>''')
    elif cfg.get('shape','square')=='triangle':
        visual_path, collision_paths=meshes(Path(path).parent/'shape_meshes',cfg,bh)
        def mesh_geom(p):
            return f'<geometry><mesh><uri>file://{p.resolve()}</uri></mesh></geometry>'
        collisions=''.join(f'''<collision name="collision_{i}">{mesh_geom(p)}
          <surface><friction><ode><mu>0.45</mu><mu2>0.45</mu2></ode></friction></surface>
          </collision>''' for i,p in enumerate(collision_paths))
        geom=mesh_geom(visual_path)
        # Approximate mass moments of the same-sized planar shape, about its center.
        iz=mass*cfg.get('shape_radius',.35)**2*.25
        ix=iy=(iz+mass*bh*bh/6)/2
        pieces.append(f'''<model name="crate"><pose>{bx} {by} {bh/2+.002} 0 0 0</pose>
          <link name="body"><inertial><mass>{mass}</mass><inertia>
          <ixx>{ix}</ixx><iyy>{iy}</iyy><izz>{iz}</izz></inertia></inertial>
          {collisions}<visual name="visual">{geom}{material('0.95 0.59 0.12 1')}</visual>
          <visual name="angle_marker"><pose>0 0 {bh/2+.003} 0 0 0</pose>
          {box_geom(.025,.25,.003)}{material('0.3 0.12 0.08 1')}</visual>
          </link></model>''')
    else:
        pieces.append(f'''<model name="crate"><pose>{bx} {by} {bh/2+.002} 0 0 0</pose>
      <link name="body"><inertial><mass>{mass}</mass><inertia>
      <ixx>{ix}</ixx><iyy>{ix}</iyy><izz>{iz}</izz></inertia></inertial>
      <collision name="collision">{geom}<surface><friction><ode><mu>0.45</mu><mu2>0.45</mu2>
      </ode></friction></surface></collision>
      <visual name="visual">{geom}{material('0.95 0.59 0.12 1')}</visual>
      <visual name="angle_marker"><pose>{size*.22} 0 {bh/2+.002} 0 0 0</pose>
      {box_geom(size*.1,size*.8,.003)}{material('0.3 0.12 0.08 1')}</visual>
      </link></model>''')
    mass=5.; ix=mass*(3*radius*radius+rh*rh)/12; iz=mass*radius*radius/2
    geom=f'<geometry><cylinder><radius>{radius}</radius><length>{rh}</length></cylinder></geometry>'
    pieces.append(f'''<model name="robot"><pose>{rx} {ry} {rh/2+.002} 0 0 0</pose>
      <link name="body"><inertial><mass>{mass}</mass><inertia>
      <ixx>{ix}</ixx><iyy>{ix}</iyy><izz>{iz}</izz></inertia></inertial>
      <collision name="collision">{geom}<surface><friction><ode><mu>0.03</mu><mu2>0.03</mu2>
      </ode></friction></surface></collision>
      <visual name="visual">{geom}{material('0.08 0.38 0.9 1')}</visual>
      <visual name="cap"><pose>0 0 {rh/2+.01} 0 0 0</pose>
      <geometry><sphere><radius>{radius*.45}</radius></sphere></geometry>
      {material('0.7 0.92 1 1')}</visual></link></model>''')
    if camera:
        altitude=max(w/2,h/2*4/3)/math.tan(1.05/2)*1.08
        pieces.append(f'''<plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
        <render_engine>ogre2</render_engine></plugin>
        <model name="camera_rig"><static>true</static><pose>0 0 {altitude} 0 1.57079632679 0</pose>
          <link name="link"><sensor name="overview" type="camera">
            <always_on>true</always_on><update_rate>10</update_rate><topic>/rs/image</topic>
            <camera><horizontal_fov>1.05</horizontal_fov>
              <image><width>640</width><height>480</height><format>R8G8B8</format></image>
              <clip><near>0.1</near><far>100</far></clip></camera>
            <plugin filename="gz-sim-camera-video-recorder-system" name="gz::sim::systems::CameraVideoRecorder">
              <service>/rs/record</service><use_sim_time>true</use_sim_time>
              <fps>10</fps><bitrate>2500000</bitrate>
            </plugin></sensor></link></model>''')
    pieces.append('</world></sdf>')
    Path(path).write_text('\n'.join(pieces))

