"""
Generative agent for creating improvised notes based on human player's input
Uses sophisticated crossover to generate novel variations
"""
import numpy as np
import random
import time
from pythonosc.udp_client import SimpleUDPClient


class GenerativeAgent:
    def __init__(self, hotness=0.0):
        """
        Initialize the generative agent
        
        Args:
            hotness: Controls variation intensity (0.0-1.0)
                     0.0 = heavily biased toward input notes
                     1.0 = uniform distribution across all values
        """
        self.hotness = hotness
        self.POINT_INTERVAL = 0.02  # Generate points every 20ms
        self.last_phrase_notes = []  # Track notes from last phrase for context
        # Create duplicate OSC client for port 5007
        try:
            self.duplicate_osc_client = SimpleUDPClient("127.0.0.1", 5007)
        except Exception as e:
            print(f"[AGENT] Warning: Could not create duplicate OSC client for port 5007: {e}")
            self.duplicate_osc_client = None
    
    def set_hotness(self, hotness):
        """
        Update the hotness parameter dynamically.
        
        Args:
            hotness: Controls variation intensity (0.0-1.0)
                     0.0 = heavily biased toward input notes
                     1.0 = uniform distribution across all values
        """
        self.hotness = max(0.0, min(1.0, hotness))  # Clamp to [0, 1]
    
    def _interpolate_value(self, val1, val2):
        
        if self.hotness < np.random.random():
            blend = 0.0 if np.random.random() < 0.5 else 1.0
        else:
            blend = np.random.random()
        return val1 + blend * (val2 - val1)
    
    def _interpolate_vector(self, vec1, vec2):
        """
        Interpolate between two vectors with same logic as _interpolate_value
        
        Returns: Interpolated vector
        """
        if self.hotness < np.random.random():
            blend_factor = 0.0 if np.random.random() < 0.5 else 1.0
        else:
            blend_factor = np.random.random()
        return [v1 + blend_factor * (v2 - v1) for v1, v2 in zip(vec1, vec2)]
    
    def _mutate(self):
        if 1 / ((1-self.hotness) * 985 + 15) > np.random.random(): #mutation chance based on hotness
           return True
        else:
           return False
    
    def crossover(self, note1, note2):
        """
        Vector-based crossover between two notes with hotness-influenced distribution.
        
        Args:
            note1: First note with 'data_points', 'fingers', 'pause_after'
            note2: Second note with 'data_points', 'fingers', 'pause_after'
        
        Returns: New note with same structure (points collection + metadata)
        """
        note1_points = note1['data_points']
        note2_points = note2['data_points']
        params_per_point = len(note1_points[0])  # 5: x, y, z, angle, velocity

        # Set fingers (combination of both notes' fingers)
        new_fingers = []
        if self._mutate():
             new_fingers = sorted(random.sample(range(1, 5), random.randint(1, 4)))  # Random fingers
        else:
            if np.random.random() < self.hotness:  # Do combination
                all_fingers = set(note1['fingers']) | set(note2['fingers'])
                new_fingers = sorted([f for f in all_fingers if np.random.random() < 0.5])
                if not new_fingers:  # Ensure at least one finger
                    new_fingers = [np.random.choice(list(all_fingers))]
            else:  # Pick one parent entirely (50/50)
                new_fingers = note1['fingers'] if np.random.random() < 0.5 else note2['fingers']                                                            
        
        # Set first point
        new_first_point = []
        for param_idx in range(params_per_point):
            if self._mutate():
                new_first_point.append(np.random.random())  # Random value for this parameter
            else:
                val1 = note1_points[0][param_idx]
                val2 = note2_points[0][param_idx]
                new_val = self._interpolate_value(val1, val2)
                new_val = np.clip(new_val, 0, 1)
                new_first_point.append(new_val)
        
        # Set number of points
        if self._mutate():
            # upper bound derived from sampling frequency and desired max duration
            max_points = int((1.0 / self.POINT_INTERVAL) * 8)
            num_new_points = random.randint(2, max(2, max_points))  # Random number of points
        else:
            num_points1 = len(note1_points)
            num_points2 = len(note2_points)
            num_new_points = int(self._interpolate_value(float(num_points1), float(num_points2)))
            num_new_points = max(2, num_new_points)  # At least 2 points
        
        # Set rest of the points 
        new_data_points = [tuple(new_first_point)]
        current_point = list(new_first_point)
        
        for point_idx in range(1, num_new_points):
            # Calculate vectors from consecutive points with even distribution
            # Map current point index to evenly distributed source indices
            if self._mutate():
                # Random vector for mutation
                interpolated_vec = [np.random.uniform(-0.3, 0.3) for _ in range(params_per_point)]
            else:
                source_idx1 = int((point_idx / float(num_new_points)) * len(note1_points))
                source_idx1 = min(source_idx1, len(note1_points) - 1)
                if source_idx1 == 0:
                    source_idx1 = 1
                vec1 = [note1_points[source_idx1][i] - note1_points[source_idx1 - 1][i]
                        for i in range(params_per_point)]

                source_idx2 = int((point_idx / float(num_new_points)) * len(note2_points))
                source_idx2 = min(source_idx2, len(note2_points) - 1)
                if source_idx2 == 0:
                    source_idx2 = 1
                vec2 = [note2_points[source_idx2][i] - note2_points[source_idx2 - 1][i]
                        for i in range(params_per_point)]

                # Interpolate the vectors
                interpolated_vec = self._interpolate_vector(vec1, vec2)
            
            # Generate next point by applying vector to current point
            next_point = [current_point[i] + interpolated_vec[i] for i in range(params_per_point)]
            next_point = [np.clip(val, 0, 1) for val in next_point]
            
            new_data_points.append(tuple(next_point))
            current_point = next_point

        # Set pause
        if self._mutate():
            new_pause = np.random.random() * 5  # Random pause between 0 and 5 seconds
        else:
            pause1 = note1.get('pause_after', 0)
            pause2 = note2.get('pause_after', 0)
            new_pause = self._interpolate_value(pause1, pause2)
        
        # ====== STEP 5: CREATE NEW NOTE WITH SAME STRUCTURE AS HUMAN NOTES ======
        # Calculate duration based on 0.2ms interval between points
        duration = (len(new_data_points) - 1) * self.POINT_INTERVAL if len(new_data_points) > 1 else self.POINT_INTERVAL
        
        new_note = {
            'fingers': new_fingers,
            'pause_after': new_pause,
            'duration': duration,
            'data_points': new_data_points,
            'source': 'ai'
        }

        #print(f"[AGENT] Crossover created new note: fingers={new_fingers}, pause={new_pause:.2f}s, duration={duration:.2f}s, points={len(new_data_points)}")
        
        return new_note
    
    def generate_crossovers(self, note1, note2, num_crossovers=6):
        """
        Generate multiple crossovers from two notes
        
        Args:
            note1: First source note
            note2: Second source note
            num_crossovers: How many crossovers to generate
        
        Returns: List of generated notes
        """
        crossovers = []
        for i in range(num_crossovers):
            crossover = self.crossover(note1, note2)
            crossovers.append(crossover)
        return crossovers
    
    def select_notes(self, all_notes):
        """
        Select two notes for crossover based on hotness preference.
        Each note is independently selected: Low hotness favors human notes,
        high hotness favors AI notes. This allows crossovers between human and AI.
        
        Args:
            all_notes: List of all notes (human and AI) with 'source' and 'phrase' fields
        
        Returns: Tuple of (note1, note2)
        """
        if not all_notes:
            print("[AGENT] ERROR: No notes available to select from!")
            return None
            
        if len(all_notes) < 2:
            # Fallback: return same note twice if not enough notes
            if len(all_notes) == 1:
                return (all_notes[0], all_notes[0])
            return None
        
        def select_single_note():
            """Helper to select one note based on hotness"""
            use_human = not self._mutate()  # Low hotness → True, high hotness → False
            
            if use_human:
                # Filter human notes
                human_notes = [n for n in all_notes if n.get('source') == 'human']
                if not human_notes:
                    human_notes = all_notes  # Fallback to all notes if no human notes
                
                if not human_notes:
                    print("[AGENT] ERROR: No human notes and fallback empty!")
                    return all_notes[0] if all_notes else None
                
                # Weight recent phrases higher (exponential decay from latest)
                if human_notes and 'phrase' in human_notes[0]:
                    max_phrase = max(n.get('phrase', 1) for n in human_notes)
                    weights = [2 ** (n.get('phrase', 1) - max_phrase) for n in human_notes]
                    total_weight = sum(weights)
                    probabilities = [w / total_weight for w in weights]
                    return np.random.choice(human_notes, p=probabilities)
                else:
                    return random.choice(human_notes) if human_notes else all_notes[0]
            else:
                # Use AI notes
                ai_notes = [n for n in all_notes if n.get('source') == 'ai']
                if not ai_notes:
                    ai_notes = all_notes  # Fallback to all notes if no AI notes
                
                if not ai_notes:
                    return all_notes[0] if all_notes else None
                    
                return random.choice(ai_notes)
        
        # Select each note independently
        try:
            note1 = select_single_note()
            note2 = select_single_note()
            
            if note1 is None or note2 is None:
                print("[AGENT] ERROR: Failed to select notes")
                return None
            
            return (note1, note2)
        except Exception as e:
            print(f"[AGENT] Error in select_notes: {e}")
            return None
    
    def generate_phrase(self, all_notes, last_phrase_num):
        """
        Generate a new phrase with note count based on last phrase size.
        Formula: n + random(-1, 1) * n/2, where n is last phrase note count
        
        Args:
            all_notes: List of all recorded notes (human and AI)
            last_phrase_num: The phrase number that just ended
        
        Returns: List of generated notes for the new phrase
        """
        # Count notes in last phrase
        last_phrase_notes = [n for n in all_notes if n.get('phrase') == last_phrase_num]
        n = len(last_phrase_notes)
        
        #print(f"[AGENT] Notes in last phrase #{last_phrase_num}: {n}")
        
        if n == 0:
            print("[AGENT] Warning: Last phrase has no notes, cannot generate")
            return []
        
        # Verify that notes have data_points
        valid_notes = [note for note in all_notes if note.get('data_points') and len(note['data_points']) > 0]
        if not valid_notes:
            print("[AGENT] ERROR: No valid notes with data_points found!")
            return []
        
        print(f"[AGENT] Valid notes with data_points: {len(valid_notes)}")
        
        # Calculate desired number of notes: n + random(-1, 1) * n/2
        variance = np.random.uniform(-1, 1) * (n / 2.0)
        num_new_notes = max(1, int(n + variance))  # At least 1 note
        
        print(f"[AGENT] Generating {num_new_notes} new notes")
        
        generated_notes = []
        for i in range(num_new_notes):
            # Select two notes for crossover
            note_pair = self.select_notes(valid_notes)
            if note_pair is None:
                print(f"[AGENT] Error: Cannot select notes for crossover")
                break
            
            note1, note2 = note_pair
            
            try:
                # Generate crossover
                new_note = self.crossover(note1, note2)
                generated_notes.append(new_note)
                # Don't print for every note to reduce spam
                #print(f"[AGENT]   Generated note {i+1}/{num_new_notes}")
            except Exception as e:
                print(f"[AGENT] Error generating note {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return generated_notes
    
    def play_phrase(self, notes, osc_client):
        """
        Play a sequence of notes (a phrase).
        
        Args:
            notes: List of note dicts to play in sequence
            osc_client: OSC client for sending messages
        
        Returns: Total time taken to play phrase
        """
        if not notes:
            print("[AGENT] Error: No notes to play in phrase")
            return 0.0
        
        if osc_client is None:
            print("[AGENT] Error: No OSC client provided, cannot play phrase")
            return 0.0
        
        start_time = time.time()
        
        for i, note in enumerate(notes):
            try:
                #print(f"\n[AGENT] Playing note {i+1}/{len(notes)} of phrase")
                self.play_note(note, osc_client)
            except Exception as e:
                print(f"[AGENT] Error playing note {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue

        #play empty note at end to signify end of phrase
        try:
            self.play_note({'data_points': [(0, 0, 0, 0, 0)], 'fingers': [], 'pause_after': 0, 'source': 'ai'}, osc_client)
        except Exception as e:
            print(f"[AGENT] Error playing end-of-phrase note: {e}")
            import traceback
            traceback.print_exc()
        
        total_time = time.time() - start_time
        print(f"[AGENT] Phrase complete: {len(notes)} notes in {total_time:.3f}s")
        
        return total_time
    
    def on_phrase_end(self, all_notes, last_phrase_num, osc_client, note_recorder):
        """
        Called when note_collection detects end of phrase.
        Generates and plays a new phrase automatically.
        
        Args:
            all_notes: List of all notes from note_collection.get_notes()
            last_phrase_num: The phrase number that just ended
            osc_client: OSC client for playing notes
            note_recorder: Reference to NoteRecorder to reset pause clock after playing
        """
        print(f"\n[AGENT] === GENERATING NEW PHRASE ===")
        #print(f"[AGENT] Total notes available: {len(all_notes)}")
        
        if not all_notes:
            print("[AGENT] ERROR: No notes available to generate from!")
            return
        
        # Generate new phrase
        try:
            new_phrase_notes = self.generate_phrase(all_notes, last_phrase_num)
        except Exception as e:
            print(f"[AGENT] Error in generate_phrase: {e}")
            import traceback
            traceback.print_exc()
            return
        
        if not new_phrase_notes:
            print("[AGENT] Failed to generate phrase notes")
            return
        
        print(f"[AGENT] Generated {len(new_phrase_notes)} notes, now playing...")
        
        # Play new phrase
        try:
            self.play_phrase(new_phrase_notes, osc_client)
        except Exception as e:
            print(f"[AGENT] Error in play_phrase: {e}")
            import traceback
            traceback.print_exc()
        
        # IMPORTANT: Reset pause clock after agent finishes playing
        # so the clock only starts when the human plays the next note
        if note_recorder is not None:
            note_recorder.pause_start_time = None
            note_recorder.pause_triggered = False
            print(f"[AGENT] Pause clock reset - waiting for next human note...")
    
    def play_note(self, note, osc_client):
        """
        Play a note by sending its data points via OSC messages with proper timing.
        Interpolates through points at POINT_INTERVAL (0.02s) per point.
        Blocks until note is fully played including pause_after.
        
        Args:
            note: Note dict with 'data_points', 'fingers', and 'pause_after'
            osc_client: OSC client for sending messages (pythonosc.udp_client.SimpleUDPClient)
        
        Returns: Dict with 'note_duration' and 'total_duration' (including pause_after)
        """
        fingers = note.get('fingers', [])
        data_points = note.get('data_points', [])
        pause_after = note.get('pause_after', 0.0)
        
        if not data_points:
            print("[AGENT] Error: Note has no data points")
            return {'note_duration': 0.0, 'total_duration': 0.0}
        
        # Send fingers once at start (same as hand_detection)
        # Convert to string to avoid pythonosc type issues with numpy types
        fingers_str = ",".join(map(str, fingers)) if fingers else ""
        osc_client.send_message("/fingers", fingers_str)
        if self.duplicate_osc_client:
            self.duplicate_osc_client.send_message("/fingers", fingers_str)
        
        start_time = time.time()
        
        # Iterate through points with timing
        for i, point in enumerate(data_points):
            # Sleep between points (except for first point)
            if i > 0:
                time.sleep(self.POINT_INTERVAL)
            
            # Extract parameters from point tuple: (x, y, z, angle, velocity, [relative_time])
            x = float(point[0])
            y = float(point[1])
            z = float(point[2])
            angle = float(point[3])
            velocity = float(point[4])
            
            # Send OSC messages to primary client
            osc_client.send_message("/x", x)
            osc_client.send_message("/y", y)
            osc_client.send_message("/z", z)
            osc_client.send_message("/angle", angle)
            osc_client.send_message("/velocity", velocity)
            
            # Send duplicate messages to port 5007
            if self.duplicate_osc_client:
                self.duplicate_osc_client.send_message("/x", x)
                self.duplicate_osc_client.send_message("/y", y)
                self.duplicate_osc_client.send_message("/z", z)
                self.duplicate_osc_client.send_message("/angle", angle)
                self.duplicate_osc_client.send_message("/velocity", velocity)
        
        note_duration = time.time() - start_time
        
        # Apply pause_after
        if pause_after > 0:
            time.sleep(pause_after)
        
        total_duration = time.time() - start_time
        
        # Output completion message (minimal)
        #print(f"[AGENT] Note finished: {note_duration:.3f}s + {pause_after:.3f}s pause")
        
        return {
            'note_duration': note_duration,
            'pause_after': pause_after,
            'total_duration': total_duration
        }


