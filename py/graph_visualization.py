"""
Graph visualization module
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_agent_notes(agent_notes, human_notes):
    """Plot human notes and generated agent notes in same window with page switching"""
    if not agent_notes:
        print("No agent notes generated.")
        return
    
    print(f"\nDisplaying {len(human_notes)} HUMAN NOTES and {len(agent_notes)} AGENT-GENERATED NOTES\n")
    
    # Parameter names and colors
    parameter_names = ['x', 'y', 'z', 'angle', 'velocity']
    param_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    finger_names = {1: 'Index', 2: 'Middle', 3: 'Ring', 4: 'Pinky'}
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('HUMAN NOTES vs AGENT CROSSOVERS (Use LEFT/RIGHT arrows to switch pages)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Store page data
    pages_data = [
        {'title': 'HUMAN NOTES (Page 1)', 'notes': human_notes, 'is_human': True},
        {'title': 'AGENT CROSSOVERS (Page 2)', 'notes': agent_notes, 'is_human': False}
    ]
    
    # Page navigation
    page_state = {'current_page': 0}
    
    def plot_page(page_idx):
        """Plot a specific page"""
        plt.clf()
        
        # Recreate title
        fig.suptitle('HUMAN NOTES vs AGENT CROSSOVERS (Use LEFT/RIGHT arrows to switch pages)', 
                     fontsize=14, fontweight='bold', y=0.98)
        
        page_data = pages_data[page_idx]
        notes = page_data['notes']
        is_human = page_data['is_human']
        
        # Calculate grid layout
        num_notes = len(notes)
        cols = 2
        rows = (num_notes + cols - 1) // cols
        
        # Plot notes
        for note_idx, note in enumerate(notes, 1):
            ax = fig.add_subplot(rows, cols, note_idx)
            
            # Plot trajectory for both human and agent notes
            times = []
            values_by_param = {i: [] for i in range(5)}
            
            if is_human:
                # Human notes: use relative_time from data_points
                for i, data_point in enumerate(note['data_points']):
                    times.append(data_point[5])  # relative_time at index 5
                    for param_idx in range(5):
                        values_by_param[param_idx].append(data_point[param_idx])
            else:
                # Agent notes: generate time array for data points
                data_points = note['data_points']
                for i in range(len(data_points)):
                    times.append(i * 0.033)  # Approximate time (30fps)
                    for param_idx in range(5):
                        values_by_param[param_idx].append(data_points[i][param_idx])
            
            # Plot each parameter as a line
            for param_idx, param_name in enumerate(parameter_names):
                ax.plot(times, values_by_param[param_idx], 'o-', 
                       color=param_colors[param_idx], label=param_name, linewidth=2, markersize=4)
            
            ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.legend(loc='best', fontsize=9)
            
            ax.grid(True, alpha=0.3)
            
            # Create title
            fingers_str = ', '.join([finger_names.get(f, f'F{f}') for f in note['fingers']])
            
            if is_human:
                phrase = note.get('phrase', 1)
                pause_after = note.get('pause_after', 0)
                title = f"Phrase {phrase} - Note {note_idx}: {fingers_str}"
                if pause_after > 0:
                    title += f" | Pause: {pause_after:.2f}s"
            else:
                title = f"Crossover {note_idx}: {fingers_str} | Duration: {note['duration']:.2f}s"
            
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    def on_key(event):
        """Handle keyboard navigation"""
        if event.key == 'left':
            page_state['current_page'] = (page_state['current_page'] - 1) % len(pages_data)
            plot_page(page_state['current_page'])
            fig.canvas.draw()
        elif event.key == 'right':
            page_state['current_page'] = (page_state['current_page'] + 1) % len(pages_data)
            plot_page(page_state['current_page'])
            fig.canvas.draw()
    
    # Connect keyboard event
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Plot first page
    plot_page(0)
    plt.show()