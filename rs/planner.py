"""Single-crate Sokoban BFS and collision-aware grid navigation. No LLM."""
from collections import deque
from dataclasses import dataclass
import numpy as np
from .shapes import clearance

DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
def add(a, b): return (a[0]+b[0], a[1]+b[1])

@dataclass
class Board:
    rows: int
    cols: int
    walls: set
    robot: tuple
    box: tuple
    goal: tuple
    cell: float = .8

    def free(self, p):
        return 0 <= p[0] < self.rows and 0 <= p[1] < self.cols and p not in self.walls

    def xy(self, p):
        return np.array([(p[1]-(self.cols-1)/2)*self.cell,
                         ((self.rows-1)/2-p[0])*self.cell], dtype=float)

    def cell_at(self, xy):
        return (int(np.rint((self.rows-1)/2-xy[1]/self.cell)),
                int(np.rint(xy[0]/self.cell+(self.cols-1)/2)))

    def validate(self):
        assert self.rows >= 7 and self.cols >= 7, '最初は7×7以上にしてください（外壁を含む）。'
        assert all(self.free(p) for p in (self.robot, self.box, self.goal))
        assert self.robot != self.box
        for r in range(self.rows):
            assert (r,0) in self.walls and (r,self.cols-1) in self.walls
        for c in range(self.cols):
            assert (0,c) in self.walls and (self.rows-1,c) in self.walls

    def text(self):
        out=[]
        for r in range(self.rows):
            out.append(''.join('#' if (r,c) in self.walls else
                '*' if (r,c)==self.box==self.goal else
                '$' if (r,c)==self.box else '@' if (r,c)==self.robot else
                '.' if (r,c)==self.goal else ' ' for c in range(self.cols)))
        return '\n'.join(out)

def make_board(cfg, case='turn'):
    if case == 'maze_complex':
        rows = cols = 9
        walls = {(r,c) for r in range(rows) for c in range(cols)
                 if r in (0,rows-1) or c in (0,cols-1)}
        walls.update({(4,3), (4,4), (4,5), (5,4), (5,5)})
        board = Board(rows,cols,walls,(6,2),(6,3),(2,5),cfg['cell'])
        board.validate()
        return board
    rows, cols = cfg['rows'], cfg['cols']
    walls={(r,c) for r in range(rows) for c in range(cols)
           if r in (0,rows-1) or c in (0,cols-1)}
    # Robot can get around the box at turns; no impossible tight elbow.
    robot, box = (rows-3,1), (rows-3,2)
    goal = (rows-3,cols-3) if case == 'straight' else (2,cols-3)
    if case in ('turn','maze'):
        walls.add((2,2))
    if case == 'maze':
        walls.add((rows-4,cols-4))
        goal=(2,cols-4)
    if case == 'train':
        robot, box, goal=(rows//2,cols//2-1),(rows//2,cols//2),(rows//2,cols//2+1)
    if case not in ('train','straight','turn','maze'): raise ValueError(case)
    board=Board(rows,cols,walls,robot,box,goal,cfg['cell'])
    board.validate()
    return board

def solve(board, robot=None, box=None):
    """BFS over (robot cell, box cell). Returns (direction, is_push) moves."""
    start=(robot or board.robot, box or board.box)
    q=deque([start]); prev={start:None}
    while q:
        state=q.popleft(); r,b=state
        if b == board.goal:
            moves=[]
            while prev[state] is not None:
                parent,d,push=prev[state]; moves.append((d,push)); state=parent
            return moves[::-1]
        for d in DIRECTIONS:
            nr=add(r,d)
            if not board.free(nr): continue
            push=nr==b; nb=add(b,d) if push else b
            if not board.free(nb): continue
            nxt=(nr,nb)
            if nxt not in prev:
                prev[nxt]=(state,d,push); q.append(nxt)
    raise ValueError('この盤面・現在状態からの解がありません。')

def push_plan(board):
    r,b=board.robot,board.box; result=[]
    for d,push in solve(board):
        if push: result.append({'box':b,'behind':r,'target':add(b,d),'direction':d}); b=add(b,d)
        r=add(r,d)
    return result

def robot_path(board, start, target, box_xy, box_yaw, box_size, radius, cfg=None):
    """BFS with sampled inflated-obstacle checks (not just cell occupancy)."""
    def clear(p):
        for w in board.walls:
            delta=np.maximum(np.abs(p-board.xy(w))-board.cell/2,0)
            if np.linalg.norm(delta) < radius+.012: return False
        if cfg and cfg.get('shape','square')!='square':
            return clearance(cfg,box_yaw,p-box_xy) >= radius+.012
        c,s=np.cos(box_yaw),np.sin(box_yaw)
        local=np.array([[c,s],[-s,c]]) @ (p-box_xy)
        delta=np.maximum(np.abs(local)-box_size/2,0)
        return np.linalg.norm(delta) >= radius+.012
    def edge(a,b): return all(clear(p) for p in np.linspace(a,b,21))
    source=board.cell_at(start)
    candidates=[p for p in (source,)+tuple(add(source,d) for d in DIRECTIONS)
                if board.free(p) and edge(start,board.xy(p))]
    q=deque(candidates); prev={p:None for p in candidates}
    while q:
        p=q.popleft()
        if p==target:
            path=[]
            while p is not None: path.append(board.xy(p)); p=prev[p]
            return path[::-1]
        for d in DIRECTIONS:
            n=add(p,d)
            if board.free(n) and n not in prev and edge(board.xy(p),board.xy(n)):
                prev[n]=p; q.append(n)
    raise ValueError('物理的な余白を確保した回り込み経路がありません。')
