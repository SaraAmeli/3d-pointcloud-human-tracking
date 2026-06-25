# Kalman filter + tracker
from scipy.optimize import linear_sum_assignment

class KalmanTrack:
    def __init__(self, tid, pos, ts):
        self.id = tid
        self.x = np.zeros((6,1))
        self.x[0:3,0] = np.array(pos)
        self.P = np.eye(6)*0.1
        self.last_ts = ts
        self.hits = 1
        self.time_since_update = 0
        self.history = [(pos, ts)]
        self.smoothed_speed = 0.0

    def predict(self, ts):
        dt = max(1e-6, (ts - self.last_ts)/1000.0)
        F = np.eye(6)
        F[0,3] = F[1,4] = F[2,5] = dt
        Q = np.eye(6)*1e-2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.last_ts = ts
        self.time_since_update += 1

    def update(self, pos, ts):
        z = np.array(pos).reshape((3,1))
        H = np.zeros((3,6)); H[0,0]=H[1,1]=H[2,2]=1
        R = np.eye(3)*1e-2
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P
        # update velocity
        prev_pos, prev_ts = self.history[-1]
        dt = max(1e-6, (ts - prev_ts)/1000.0)
        inst_v = (np.array(pos) - np.array(prev_pos))/dt
        alpha = 0.6
        self.x[3:6,0] = alpha*inst_v + (1-alpha)*self.x[3:6,0]
        speed = np.linalg.norm(self.x[3:6,0])
        self.smoothed_speed = 0.6*speed + 0.4*self.smoothed_speed
        self.history.append((pos, ts))
        self.hits += 1
        self.time_since_update = 0
        self.last_ts = ts

    def state(self):
        pos = self.x[0:3,0].tolist()
        vel = self.x[3:6,0].tolist()
        speed = float(np.linalg.norm(self.x[3:6,0]))
        return pos, vel, speed

class SimpleTracker:
    def __init__(self, max_distance=1.0, max_age=30):
        self.tracks = {}
        self.next_id = 1
        self.max_distance = max_distance
        self.max_age = max_age

    def step(self, detections, ts):
        # predict
        for t in list(self.tracks.values()):
            t.predict(ts)
        dets = np.array([d['centroid'] for d in detections]) if len(detections)>0 else np.empty((0,3))
        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[tid].x[0:3,0] for tid in track_ids]) if len(track_ids)>0 else np.empty((0,3))
        assignments = []
        unmatched_dets = list(range(len(dets)))
        unmatched_tracks = track_ids.copy()
        if dets.shape[0]>0 and track_centroids.shape[0]>0:
            cost = np.linalg.norm(track_centroids[:,None,:] - dets[None,:,:], axis=2)
            r,c = linear_sum_assignment(cost)
            for rr,cc in zip(r,c):
                if cost[rr,cc]<=self.max_distance:
                    tid = track_ids[rr]
                    assignments.append((tid,cc))
                    if cc in unmatched_dets: unmatched_dets.remove(cc)
                    if tid in unmatched_tracks: unmatched_tracks.remove(tid)
        # update matched
        for tid,didx in assignments:
            self.tracks[tid].update(dets[didx].tolist(), ts)
        # add new tracks
        for didx in unmatched_dets:
            pos = dets[didx].tolist()
            self.tracks[self.next_id] = KalmanTrack(self.next_id,pos,ts)
            self.next_id += 1
        # drop old
        drop = [tid for tid,tr in self.tracks.items() if tr.time_since_update>self.max_age]
        for tid in drop: del self.tracks[tid]
        # output
        outs = []
        for tid,tr in self.tracks.items():
            pos, vel, speed = tr.state()
            conf = max(0.0, 1.0 - (tr.time_since_update / float(self.max_age+1)))
            outs.append({'human_id': int(tr.id), 'centroid': [float(v) for v in pos],
                         'velocity':[float(v) for v in vel], 'speed':float(speed), 'confidence':float(conf)})
        return outs
