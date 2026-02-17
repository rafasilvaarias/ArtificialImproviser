import cv2
import numpy as np
import time
import threading
from pythonosc import udp_client

# Import modules
from hand_detection import (
    HandDetector, get_touching_fingers, get_finger_colors, 
    get_hand_position, get_hand_angle, draw_hand_landmarks
)
from note_collection import NoteRecorder
from graph_visualization import plot_agent_notes
from generative_agent import GenerativeAgent

# ============================================================================
# GLOBAL CONFIGURATION - Control here
# ============================================================================
ENABLE_NOTE_COLLECTION = True  # Toggle note recording on/off
ENABLE_GRAPH_VISUALIZATION = True  # Toggle graph display on/off
ENABLE_OSC = True  # Toggle OSC messages on/off
ENABLE_GENERATIVE_AGENT = True  # Toggle generative agent on/off

OSC_IP = "127.0.0.1"  # localhost
OSC_PORT = 5005  # Default Max/MSP port

# ============================================================================

# Initialize hand detector
detector = HandDetector(model_path='hand_landmarker.task')

# Camera setup
videoCap = cv2.VideoCapture(0)

# OSC setup
if ENABLE_OSC:
    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    osc_client_ai = udp_client.SimpleUDPClient(OSC_IP, 5006)  # AI agent on port 5006

# Generative agent system
if ENABLE_GENERATIVE_AGENT:
    agent = GenerativeAgent(hotness=0)  # Low hotness = tight clustering to input notes
else:
    agent = None

# Flag to prevent overlapping agent generations
agent_is_playing = False
if ENABLE_NOTE_COLLECTION:
    note_recorder = NoteRecorder(agent=agent, enable_agent=ENABLE_GENERATIVE_AGENT)

# Track touching state with frame counter for debouncing
finger_tracking = {}  # {hand_idx: {finger_num: frame_count}}
debounce_frames = 20 
last_print_time = time.time()
print_interval = 1.0  # Print every 1 second

# Track hand position for velocity calculation
previous_hand_pos = None
previous_time = time.time()
previous_velocity = 0.0  # For velocity smoothing
previous_active_fingers = []  # Track previous frame's active fingers

# Session start time (for all timestamps)
session_start_time = time.time()

# Continuous loop for video feed
while True:
    success, img = videoCap.read()
    
    if not success:
        print("Failed to read from camera")
        break
    
    h, w, c = img.shape
    
    # Detect hands
    results = detector.detect(img)
    
    current_time = time.time()
    should_print = (current_time - last_print_time) >= print_interval
    
    # Draw landmarks if hands detected
    if results.hand_landmarks:
        # Only use the first detected hand
        hand_landmarks = results.hand_landmarks[0]
        hand_idx = 0
        
        # Get hand metrics first (for z distance)
        hand_pos = get_hand_position(hand_landmarks)
        hand_angle = get_hand_angle(hand_landmarks)
        
        # Get touching fingers (z-adjusted)
        touching = get_touching_fingers(hand_landmarks, hand_pos[2])
        
        # Initialize hand tracking if needed
        if hand_idx not in finger_tracking:
            finger_tracking[hand_idx] = {}
        
        # Update frame counters for currently detected fingers
        current_fingers = set(touching)
        previous_fingers = set(finger_tracking[hand_idx].keys())
        
        # Add or reset counters for detected fingers
        for finger in current_fingers:
            finger_tracking[hand_idx][finger] = 0
        
        # Increment counters for fingers no longer detected
        for finger in list(finger_tracking[hand_idx].keys()):
            if finger not in current_fingers:
                finger_tracking[hand_idx][finger] += 1
                # Remove if debounce threshold exceeded
                if finger_tracking[hand_idx][finger] >= debounce_frames:
                    del finger_tracking[hand_idx][finger]
        
        # Get active fingers (those with 0 counter)
        active_fingers = [f for f, count in finger_tracking[hand_idx].items() if count == 0]
        
        # Calculate velocity
        velocity = 0.0
        if previous_hand_pos is not None:
            time_delta = current_time - previous_time
            if time_delta > 0:
                # Distance in normalized coordinates
                dx = hand_pos[0] - previous_hand_pos[0]
                dy = hand_pos[1] - previous_hand_pos[1]
                dz = hand_pos[2] - previous_hand_pos[2]
                
                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                raw_velocity = distance / time_delta
                
                # Normalize to 0-1 range (max velocity = 2.0)
                velocity = np.clip(raw_velocity / 2.0, 0, 1)
                
                # Smooth velocity: only allow change of ±0.05 per frame
                velocity = np.clip(velocity, previous_velocity - 0.05, previous_velocity + 0.05)
        
        # Update previous position and time
        previous_hand_pos = hand_pos
        previous_time = current_time
        previous_velocity = velocity
        
        # ====================================================================
        # NOTE COLLECTION
        # ====================================================================
        if ENABLE_NOTE_COLLECTION:
            # Check if fingers changed (new note or pause)
            current_time_session = current_time - session_start_time
            fingers_changed = (sorted(active_fingers) != sorted(previous_active_fingers))
            
            if active_fingers:  # Fingers are pinched
                if fingers_changed:
                    # Start a new note
                    note_recorder.start_note(active_fingers, hand_pos[0], hand_pos[1], hand_pos[2], 
                                            hand_angle, velocity, current_time_session)
                else:
                    # Continue recording current note - record every frame, but sample every 10
                    note_recorder.record_point(hand_pos[0], hand_pos[1], hand_pos[2], 
                                             hand_angle, velocity, current_time_session)
            else:  # No fingers pinched
                # Call pause every frame (not just on transition) to detect sustained pauses
                note_recorder.pause(current_time_session)
            
            previous_active_fingers = active_fingers.copy()
            
            # Check if a phrase just ended and trigger generative agent in background thread
            if ENABLE_GENERATIVE_AGENT and note_recorder.last_phrase_ended is not None and not agent_is_playing:
                phrase_num = note_recorder.last_phrase_ended
                print(f"\n[MAIN] Phrase {phrase_num} ended - triggering generative agent in background!")
                
                # Run agent in separate thread so camera doesn't freeze
                def run_agent_in_background():
                    global agent_is_playing
                    agent_is_playing = True
                    try:
                        agent.on_phrase_end(note_recorder.get_notes(), phrase_num, osc_client_ai, note_recorder)
                        print(f"[MAIN] Generative agent finished playing phrase")
                    except Exception as e:
                        print(f"[ERROR] Generative agent error: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        agent_is_playing = False  # Clear flag when done
                
                agent_thread = threading.Thread(target=run_agent_in_background, daemon=True)
                agent_thread.start()
                
                # Reset the flag so we don't trigger repeatedly
                note_recorder.last_phrase_ended = None
        
        # ====================================================================
        # OSC MESSAGES
        # ====================================================================
        if ENABLE_OSC:
            # Send OSC messages (single hand, no hand index)
            # Convert fingers to string to avoid pythonosc type issues
            fingers_str = ",".join(map(str, active_fingers)) if active_fingers else ""
            osc_client.send_message("/fingers", fingers_str)
            osc_client.send_message("/x", hand_pos[0])
            osc_client.send_message("/y", hand_pos[1])
            osc_client.send_message("/z", hand_pos[2])
            osc_client.send_message("/velocity", velocity)
            osc_client.send_message("/angle", hand_angle)
        
        # Print every second
        if should_print:
            notes_count = len(note_recorder.notes) if ENABLE_NOTE_COLLECTION else 0
            #print(f"[STATUS] Collected {notes_count} notes | Active fingers: {active_fingers if active_fingers else 'NONE'}")
            if ENABLE_NOTE_COLLECTION and note_recorder.pause_start_time is not None:
                elapsed = current_time_session - note_recorder.pause_start_time
                remaining = note_recorder.PAUSE_PHRASE_THRESHOLD - elapsed
                #print(f"[STATUS]   PAUSE: {elapsed:.2f}s elapsed, {remaining:.2f}s until phrase end")
        
        # Get colors based on thumb contact (z-adjusted)
        colors = get_finger_colors(hand_landmarks, hand_pos[2])
        
        # Draw hand landmarks
        draw_hand_landmarks(img, hand_landmarks, colors)
    
    else:
        # Clear tracking when no hands detected
        if finger_tracking:
            finger_tracking.clear()
    
    # Update print time
    if should_print:
        last_print_time = current_time
    
    # Mirror the image horizontally
    img_mirrored = cv2.flip(img, 1)
    
    # Display the mirrored image
    cv2.imshow("CamOutput", img_mirrored)
    
    # Wait 1ms and check if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
videoCap.release()
cv2.destroyAllWindows()

print("\nProgram ended.")