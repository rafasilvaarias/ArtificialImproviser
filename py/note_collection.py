import numpy as np

class NoteRecorder:
    def __init__(self, agent=None, enable_agent=False):
        self.notes = []
        self.current_note = None
        self.last_record_time = None
        self.pause_start_time = None
        self.POINT_INTERVAL = 0.0002
        self.phrase_num = 1
        self.PAUSE_PHRASE_THRESHOLD = 3.0
        self.LAST_NOTE_PAUSE = 3.0
        self.pause_triggered = False
        self.last_phrase_ended = None
        self.agent = agent
        self.enable_agent = enable_agent
        
    def start_note(self, fingers, x, y, z, angle, velocity, timestamp):
        if self.current_note is not None:
            self.save_current_note()

        if self.notes:
            last_note = self.notes[-1]
            if last_note.get('timestamps') and float(last_note.get('pause_after', 0.0)) == 0.0:
                last_end = last_note['timestamps'][-1]
                last_note['pause_after'] = max(0.0, timestamp - last_end)

        self.pause_start_time = None
        self.pause_triggered = False

        self.current_note = {
            'fingers': fingers.copy() if fingers else [],
            'start_time': timestamp,
            'data_points': [(x, y, z, angle, velocity, 0)],
            'timestamps': [timestamp],
            'source': 'human'
        }
        self.last_record_time = timestamp
    
    def record_point(self, x, y, z, angle, velocity, timestamp):
        if self.current_note is not None:
            if timestamp - self.last_record_time >= self.POINT_INTERVAL:
                relative_time = timestamp - self.current_note['start_time']
                self.current_note['data_points'].append((x, y, z, angle, velocity, relative_time))
                self.current_note['timestamps'].append(timestamp)
                self.last_record_time = timestamp
    
    def pause(self, timestamp):
        if self.current_note is not None:
            self.save_current_note()
            self.current_note = None
        
        if self.pause_start_time is None:
            self.pause_start_time = timestamp
            self.pause_triggered = False
            print(f"[DEBUG] PAUSE_START at {timestamp:.2f}s (notes collected: {len(self.notes)})")
        else:
            pause_duration = timestamp - self.pause_start_time
            if pause_duration >= self.PAUSE_PHRASE_THRESHOLD and not self.pause_triggered:
                self.end_phrase(timestamp)
                self.pause_triggered = True

    def end_phrase(self, timestamp=None):
        print(f"\n{'='*80}")
        print(f"[PHRASE_END] === END_PHRASE #{self.phrase_num} at {timestamp:.2f}s ===")
        print(f"[PHRASE_END] Total notes collected: {len(self.notes)}")
        
        assigned_any = False
        for i, note in enumerate(self.notes):
            has_phrase = 'phrase' in note
            if not has_phrase:
                note['phrase'] = self.phrase_num
                assigned_any = True
                print(f"[PHRASE_END]   Note {i}: assigned phrase {self.phrase_num}")

        if self.notes:
            self.notes[-1]['pause_after'] = self.LAST_NOTE_PAUSE
            print(f"[PHRASE_END] Set last note's pause_after = {self.LAST_NOTE_PAUSE}s")

        similarities = []

        for i in range(len(self.notes) - 1):
            note1 = self.notes[i]
            note2 = self.notes[i + 1]
            sim = self.note_similarity(note1, note2)
            similarities.append(sim)
        
        phrase_cohesion = np.mean(similarities) if similarities else 0.0
        print(f"[PHRASE_END] ////////////////////////// Cohesion {phrase_cohesion} ")

        if self.enable_agent and self.agent is not None:
            hotness = max(0.0, min(1.0, 2 * (0.8 - phrase_cohesion)))
            self.agent.set_hotness(hotness)
            print(f"[PHRASE_END] Set generative agent hotness to {hotness:.2f} based on cohesion")

        phrase_num_ended = self.phrase_num
        if assigned_any:
            self.phrase_num += 1

        self.pause_start_time = None
        self.pause_triggered = False

        self.last_phrase_ended = phrase_num_ended
        
    def save_current_note(self):
        if self.current_note is not None:
            duration = self.current_note['timestamps'][-1] - self.current_note['start_time']
            self.current_note['duration'] = duration
            
            if 'pause_after' not in self.current_note:
                self.current_note['pause_after'] = 0.0
            
            if duration >= 0.1:
                self.notes.append(self.current_note)
            
            self.current_note = None
            self.frame_count = 0
    
    def finalize(self, current_time):
        if self.current_note is not None:
            self.save_current_note()
        
        for i in range(len(self.notes) - 1):
            end_time = self.notes[i]['timestamps'][-1]
            next_start_time = self.notes[i + 1]['start_time']
            if self.notes[i].get('pause_after', 0.0) == 0.0:
                self.notes[i]['pause_after'] = next_start_time - end_time

        for note in self.notes:
            if 'source' not in note:
                note['source'] = 'human'

        print(f"\n[DEBUG] Total notes recorded: {len(self.notes)}")
        for i, note in enumerate(self.notes):
            print(f"  Note {i}: {len(note['data_points'])} points, fingers={note['fingers']}")
    
    def note_similarity(self, note1, note2):
        fingers1 = set(note1['fingers'])
        fingers2 = set(note2['fingers'])
        if fingers1 and fingers2:
            finger_sim = len(fingers1 & fingers2) / len(fingers1 | fingers2)
        else:
            finger_sim = 0.0
        
        traj1 = np.array([(x, y, z, angle, vel) for x, y, z, angle, vel, t in note1['data_points']])
        traj2 = np.array([(x, y, z, angle, vel) for x, y, z, angle, vel, t in note2['data_points']])
        
        def path_length(traj):
            diffs = np.diff(traj, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            return np.sum(distances)
        
        path1 = path_length(traj1)
        path2 = path_length(traj2)
        
        if path1 > 0 and path2 > 0:
            path_ratio = min(path1, path2) / max(path1, path2)
        else:
            path_ratio = 0.0
        
        directions1 = np.diff(traj1, axis=0)
        directions2 = np.diff(traj2, axis=0)
        
        avg_dir1 = np.mean(directions1, axis=0)
        avg_dir2 = np.mean(directions2, axis=0)
        
        norm1 = np.linalg.norm(avg_dir1)
        norm2 = np.linalg.norm(avg_dir2)
        
        if norm1 > 0 and norm2 > 0:
            direction_sim = np.dot(avg_dir1, avg_dir2) / (norm1 * norm2)
            direction_sim = (direction_sim + 1) / 2
        else:
            direction_sim = 0.0
        
        similarity = 0.3 * finger_sim + 0.3 * path_ratio + 0.4 * direction_sim
        
        return similarity
    
    def get_notes(self):
        return self.notes
    
    def clear(self):
        self.notes = []
        self.current_note = None
        self.last_record_time = None
        self.pause_start_time = None
