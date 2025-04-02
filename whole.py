import pygame
import pymunk
import sys
import math

# --- Constants ---
SCREEN_WIDTH = 1820
SCREEN_HEIGHT = 980
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
FPS = 60
GRAVITY = (0, -980)  # Negative value because Pymunk's y-axis grows upward
EDIT_MODE = 0
SIMULATION_MODE = 1
RECTANGLE = 'rectangle'
CIRCLE = 'circle'
PIN_JOINT = 'pin'
PIVOT_JOINT = 'pivot'
WELD_JOINT = 'weld'
NO_COLLISION_GROUP = 1  # Collision group for objects that shouldn't collide
CAMERA_MOVE_SPEED = 20  # Pixels per key press

class Shape:
    """
    Base class for shapes.  Provides common attributes and methods.
    """
    def __init__(self, body, shape, color):
        self.body = body
        self.shape = shape
        self.color = color
        self.mass = 10
        self.static = False
        self.elasticity = 0.4
        self.friction = 0.7

    def apply_mass(self):
        """Applies mass to the body if it's not static."""
        if not self.static:
            self.body.mass = self.mass
            # Calculate appropriate moment of inertia based on shape
            if isinstance(self.shape, pymunk.Circle):
                # For circles: moment = m * r^2 / 2
                self.body.moment = pymunk.moment_for_circle(self.mass, 0, self.shape.radius)
            elif isinstance(self.shape, pymunk.Poly):
                # For polygons like rectangles
                self.body.moment = pymunk.moment_for_poly(self.mass, self.shape.get_vertices())

    def draw(self, screen):
        """Draws the shape on the screen.  To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the draw method.")

class Rectangle(Shape):
    """
    Represents a rectangle shape.
    """
    def __init__(self, body, shape, color, width, height):
        super().__init__(body, shape, color)
        self.width = width
        self.height = height

    def draw(self, screen):
        """Draws the rectangle on the screen."""
        # Get vertices relative to the body's center of gravity
        local_verts = self.shape.get_vertices()
        # Transform local vertices to world coordinates using the body's position and angle
        world_verts = [self.body.local_to_world(v) for v in local_verts]
        # Convert world coordinates to Pygame screen coordinates
        pygame_verts = [to_pygame(v) for v in world_verts]
        # Draw the polygon using the calculated screen coordinates
        pygame.draw.polygon(screen, self.color, pygame_verts)
        # Optional: Draw outline
        # pygame.draw.polygon(screen, BLACK, pygame_verts, 1)

class Circle(Shape):
    """
    Represents a circle shape.
    """
    def __init__(self, body, shape, color, radius):
        super().__init__(body, shape, color)
        self.radius = radius

    def draw(self, screen):
        """Draws the circle on the screen."""
        position = to_pygame(self.body.position)
        pygame.draw.circle(screen, self.color, position, int(self.radius))
        angle_radians = self.body.angle
        end_point_x = position[0] + self.radius * math.cos(angle_radians)
        end_point_y = position[1] + self.radius * math.sin(angle_radians)
        pygame.draw.line(screen, BLACK, position, (end_point_x, end_point_y), 2)

class Joint:
    """
    Represents a joint between two bodies.
    """
    def __init__(self, joint_type, joint):
        self.joint_type = joint_type
        self.joint = joint
        self.body_a = joint.a
        self.body_b = joint.b

    def draw(self, screen):
        """Draws the joint on the screen.  To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the draw method.")

class PinJoint(Joint):
    """
    Represents a pin joint.
    """
    def __init__(self, joint):
        super().__init__(PIN_JOINT, joint)
        self.anchor_a = joint.anchor_a
        self.anchor_b = joint.anchor_b

    def draw(self, screen):
        """Draws the pin joint."""
        if self.body_a:
            pos_a = to_pygame(self.body_a.position + self.anchor_a)
        else:
            pos_a = to_pygame(self.anchor_a)
        if self.body_b:
            pos_b = to_pygame(self.body_b.position + self.anchor_b)
        else:
            pos_b = to_pygame(self.anchor_b)
        pygame.draw.line(screen, GREEN, pos_a, pos_b, 2)
        pygame.draw.circle(screen, RED, pos_a, 5)
        pygame.draw.circle(screen, RED, pos_b, 5)

class PivotJoint(Joint):
    """
    Represents a pivot joint. Allows rotation around a single point.
    """
    def __init__(self, joint):
        super().__init__(PIVOT_JOINT, joint)
        # Store anchor relative to body_a for consistent drawing
        # Note: Pymunk PivotJoint has anchor_a and anchor_b relative to bodies
        self.anchor_a_local = joint.anchor_a
        self.body_a_id = id(joint.a) # Store IDs for potential use
        self.body_b_id = id(joint.b)

    def draw(self, screen):
        """Draws the pivot joint."""
        # Calculate the current world position of the anchor point on body_a
        if not self.body_a:
            return # Cannot draw if body_a is missing

        current_joint_world_pos = self.body_a.local_to_world(self.anchor_a_local)
        joint_pos_pygame = to_pygame(current_joint_world_pos)

        # Draw a circle marker for the pivot point
        pygame.draw.circle(screen, RED, joint_pos_pygame, 6) # Slightly larger circle
        pygame.draw.circle(screen, BLACK, joint_pos_pygame, 6, 1) # Black outline

        # Optional: Draw lines to connected bodies' centers
        if self.body_b:
            pos_a = to_pygame(self.body_a.position)
            pos_b = to_pygame(self.body_b.position)
            pygame.draw.line(screen, GRAY, joint_pos_pygame, pos_a, 1)
            pygame.draw.line(screen, GRAY, joint_pos_pygame, pos_b, 1)

class WeldJoint(Joint):
    """
    Represents a weld joint that attaches two bodies rigidly together.
    Uses a PivotJoint and a GearJoint internally in Pymunk.
    """
    def __init__(self, pivot_joint):
        # Pass the pivot_joint to super to correctly initialize body_a and body_b
        super().__init__(WELD_JOINT, pivot_joint)
        # Store the anchor point relative to body_a's coordinate system
        self.anchor_a_local = pivot_joint.anchor_a
        # Store body IDs for reference (e.g., collision checking)
        self.body_a_id = id(pivot_joint.a)
        self.body_b_id = id(pivot_joint.b)

    def draw(self, screen):
        """Draws the weld joint."""
        # Calculate the current world position of the anchor point on body_a
        # Ensure body_a exists before attempting to use it
        if not self.body_a:
             return # Cannot draw if body_a is missing

        current_joint_world_pos = self.body_a.local_to_world(self.anchor_a_local)
        joint_pos_pygame = to_pygame(current_joint_world_pos)

        # Draw a black square at the joint's current position
        square_size = 10
        pygame.draw.rect(screen, BLACK,
                        (joint_pos_pygame[0] - square_size//2,
                         joint_pos_pygame[1] - square_size//2,
                         square_size, square_size))

        # Draw lines to connected bodies' centers (optional, but can be helpful)
        # Ensure body_b exists before attempting to use it
        if self.body_b:
            pos_a = to_pygame(self.body_a.position)
            pos_b = to_pygame(self.body_b.position)
            # Draw lines from the joint marker to the center of each body
            pygame.draw.line(screen, BLACK, joint_pos_pygame, pos_a, 1) # Thinner line
            pygame.draw.line(screen, BLACK, joint_pos_pygame, pos_b, 1) # Thinner line

def create_rectangle(x, y, width, height):
    """
    Creates a PyMunk rectangle body and shape.
    """
    body = pymunk.Body(0, 1)
    # Center the rectangle at the midpoint of the coordinates
    x_center = x + width/2
    y_center = y + height/2
    body.position = x_center, y_center
    shape = pymunk.Poly.create_box(body, (width, height))
    shape.elasticity = 0.4
    shape.friction = 0.7
    # Set default collision type
    shape.collision_type = 1
    return body, shape

def create_circle(x, y, radius):
    """
    Creates a PyMunk circle body and shape.
    """
    body = pymunk.Body(0, 1)
    body.position = x, y
    shape = pymunk.Circle(body, radius)
    shape.elasticity = 0.4
    shape.friction = 0.7
    # Set default collision type
    shape.collision_type = 1
    return body, shape

def to_pygame(p, camera_offset=(0, 0)):
    """
    Convert Pymunk coordinates to Pygame coordinates with camera offset.
    """
    if hasattr(p, 'x') and hasattr(p, 'y'):
        # It's a Vec2d or similar object
        return int(p.x - camera_offset[0]), int(-p.y + SCREEN_HEIGHT - camera_offset[1])
    else:
        # It's a tuple (x, y)
        return int(p[0] - camera_offset[0]), int(-p[1] + SCREEN_HEIGHT - camera_offset[1])

def from_pygame(p, camera_offset=(0, 0)):
    """
    Convert Pygame coordinates to Pymunk coordinates with camera offset.
    """
    return p[0] + camera_offset[0], SCREEN_HEIGHT - p[1] + camera_offset[1]

def add_shape(space, shapes, shape_type, body, shape, color):
    """
    Adds a shape to the space and the list of shapes.
    """
    if shape_type == RECTANGLE:
        obj = Rectangle(body, shape, color, shape.bb.right - shape.bb.left, shape.bb.top - shape.bb.bottom)
    elif shape_type == CIRCLE:
        obj = Circle(body, shape, color, shape.radius)
    shapes.append(obj)
    space.add(body, shape)

def should_collide(body1, body2, connected_components):
    """Check if two bodies should collide based on joint connections."""
    body1_id = id(body1)
    body2_id = id(body2)
    
    # Check if these bodies are in the same connected component
    for component in connected_components:
        if body1_id in component and body2_id in component:
            return False
    
    return True

def add_joint(space, joints, joint_type, body_a, body_b, anchor_point_world, connected_components):
    """
    Adds a joint to the space and the list of joints.
    Handles creation of appropriate Pymunk constraints based on type.
    """
    j_wrapper = None # The wrapper object for our internal list
    body_a_id = id(body_a) if body_a else None
    body_b_id = id(body_b) if body_b else None

    # Calculate anchors relative to each body's center of gravity
    # These are needed for both Weld and Pivot joints
    anchor1 = body_a.world_to_local(anchor_point_world)
    anchor2 = body_b.world_to_local(anchor_point_world)

    # Use very large finite values for force and bias for maximum rigidity
    max_joint_force = 1e9  # Increased for even more rigidity
    max_joint_bias = 1e8   # Increased for even more rigidity

    if joint_type == WELD_JOINT:
        # 1. PivotJoint (part of weld)
        pivot = pymunk.PivotJoint(body_a, body_b, anchor1, anchor2)
        pivot.max_force = max_joint_force
        pivot.max_bias = max_joint_bias
        pivot.error_bias = 0
        space.add(pivot)

        # 2. GearJoint (part of weld) - locks relative rotation
        # Using ratio=1.0 means bodies should maintain the same relative angle
        initial_phase = body_b.angle - body_a.angle
        gear = pymunk.GearJoint(body_a, body_b, initial_phase, 1.0)
        gear.max_force = max_joint_force * 10  # Even higher for rotation locking
        gear.max_bias = max_joint_bias * 10    # Even higher for rotation locking
        gear.error_bias = 0
        space.add(gear)

        j_wrapper = WeldJoint(pivot) # Wrapper uses the pivot part
        j_wrapper.gear_joint = gear  # Store gear joint for potential removal later

    elif joint_type == PIVOT_JOINT:
        # Create a single PivotJoint constraint with high force/bias settings
        pivot = pymunk.PivotJoint(body_a, body_b, anchor1, anchor2)
        # Set high force/bias for position stability while allowing rotation
        pivot.max_force = max_joint_force  # Same high value as weld joint
        pivot.max_bias = max_joint_bias    # Same high value as weld joint
        pivot.error_bias = 0
        
        # Add slide joint with min=max=distance to maintain rigid distance without restricting rotation
        # This creates a fixed distance constraint that prevents stretching
        current_distance = (body_a.position + anchor1 - body_b.position - anchor2).length
        slide = pymunk.SlideJoint(body_a, body_b, anchor1, anchor2, current_distance, current_distance)
        slide.max_force = max_joint_force
        slide.max_bias = max_joint_bias
        slide.error_bias = 0
        
        space.add(pivot, slide)
        j_wrapper = PivotJoint(pivot) # Use the PivotJoint wrapper
        
        # Store the slide joint reference in the PivotJoint wrapper for potential removal later
        j_wrapper.slide_joint = slide

    # Update connected components logic
    if body_a and body_b:
        body_a_id = id(body_a)
        body_b_id = id(body_b)
        
        # Find which component each body belongs to
        comp_a = None
        comp_b = None
        comp_a_idx = -1
        comp_b_idx = -1
        
        for i, component in enumerate(connected_components):
            if body_a_id in component:
                comp_a = component
                comp_a_idx = i
            if body_b_id in component:
                comp_b = component
                comp_b_idx = i
        
        # Case 1: Both bodies are already in the same component
        if comp_a is not None and comp_a is comp_b:
            pass  # Nothing to do
            
        # Case 2: Body A is in a component, B is not
        elif comp_a is not None and comp_b is None:
            connected_components[comp_a_idx].add(body_b_id)
            
        # Case 3: Body B is in a component, A is not
        elif comp_a is None and comp_b is not None:
            connected_components[comp_b_idx].add(body_a_id)
            
        # Case 4: Both bodies are in different components
        elif comp_a is not None and comp_b is not None:
            # Merge components
            merged_component = comp_a.union(comp_b)
            connected_components[comp_a_idx] = merged_component
            # Remove the other component
            connected_components.pop(comp_b_idx)
            
        # Case 5: Neither body is in a component
        else:
            # Create a new component
            connected_components.append({body_a_id, body_b_id})
            
        print(f"Updated connected components for {joint_type} joint. Total components: {len(connected_components)}")

    if j_wrapper:
        joints.append(j_wrapper)

def draw_ui(screen, game_mode, start_button_rect, start_button_text, stop_button_text,
            weld_joint_button_rect, weld_joint_button_text,
            pivot_joint_button_rect, pivot_joint_button_text,
            rect_button_rect, rect_button_text,
            circle_button_rect, circle_button_text,
            adding_joint, current_joint_type, camera_offset=(0, 0)):
    """
    Draws the UI elements. UI elements are in screen space, not world space,
    so they are not affected by camera position.
    """
    # Top Menu Bar (Shapes, Start/Stop)
    pygame.draw.rect(screen, GRAY, (0, 0, SCREEN_WIDTH, 40))
    pygame.draw.rect(screen, GREEN if game_mode == EDIT_MODE else RED, start_button_rect, 0) # Filled button
    start_stop_text = start_button_text if game_mode == EDIT_MODE else stop_button_text
    text_rect = start_stop_text.get_rect(center=start_button_rect.center)
    screen.blit(start_stop_text, text_rect)

    # Shape Buttons
    pygame.draw.rect(screen, BLUE, rect_button_rect, 0)
    text_rect = rect_button_text.get_rect(center=rect_button_rect.center)
    screen.blit(rect_button_text, text_rect)

    pygame.draw.rect(screen, RED, circle_button_rect, 0)
    text_rect = circle_button_text.get_rect(center=circle_button_rect.center)
    screen.blit(circle_button_text, text_rect)

    # Second Menu Bar (Joints)
    pygame.draw.rect(screen, GRAY, (0, 40, SCREEN_WIDTH, 35)) # Slightly taller

    # Weld Joint Button
    pygame.draw.rect(screen, BLACK, weld_joint_button_rect, 0) # Black button
    text_rect = weld_joint_button_text.get_rect(center=weld_joint_button_rect.center)
    screen.blit(weld_joint_button_text, text_rect)

    # Pivot Joint Button
    pygame.draw.rect(screen, RED, pivot_joint_button_rect, 0) # Red button
    text_rect = pivot_joint_button_text.get_rect(center=pivot_joint_button_rect.center)
    screen.blit(pivot_joint_button_text, text_rect)

    # Highlight active joint button if adding_joint
    if adding_joint:
        highlight_rect = None
        if current_joint_type == WELD_JOINT:
            highlight_rect = weld_joint_button_rect
        elif current_joint_type == PIVOT_JOINT:
            highlight_rect = pivot_joint_button_rect
        # Add elif for PIN_JOINT if re-enabled

        if highlight_rect:
            pygame.draw.rect(screen, (255, 255, 0), highlight_rect, 3) # Yellow border

    # Display the current game mode text
    if game_mode == SIMULATION_MODE:
        sim_text_surf = font.render("Running Simulation", True, RED)
        screen.blit(sim_text_surf, (SCREEN_WIDTH / 2 - sim_text_surf.get_width() / 2, 10))

    # Optional: Display camera position for debugging
    camera_text = font.render(f"Camera: {camera_offset[0]}, {camera_offset[1]}", True, BLACK)
    screen.blit(camera_text, (SCREEN_WIDTH - 240, 75))

def handle_input(event, game_mode, selected_shape, drawing, start_pos, shapes, space,
                 adding_joint, joint_type, joints, connected_components,
                 dragging_object=None, drag_offset=None, dragged_complex=None,
                 camera_offset=(0, 0)):
    """
    Handles user input events. Modified to account for camera position.
    """
    # Default return values
    new_game_mode = game_mode
    new_selected_shape = selected_shape
    new_drawing = drawing
    new_start_pos = start_pos
    new_adding_joint = adding_joint
    new_joint_type = joint_type
    new_dragging_object = dragging_object
    new_drag_offset = drag_offset
    new_dragged_complex = dragged_complex if dragged_complex else []
    new_camera_offset = camera_offset  # Pass camera offset through

    if event.type == pygame.MOUSEBUTTONDOWN:
        pos = pygame.mouse.get_pos()
        if game_mode == EDIT_MODE:
            # --- Button Click Handling ---
            if start_button_rect.collidepoint(pos):
                new_game_mode = SIMULATION_MODE
                new_selected_shape = None
                new_drawing = False
                new_adding_joint = False
                new_dragging_object = None # Stop dragging
                new_dragged_complex = []   # Clear complex
            elif rect_button_rect.collidepoint(pos):
                new_selected_shape = RECTANGLE
                new_drawing = False
                new_adding_joint = False
                new_dragging_object = None
                new_dragged_complex = []
            elif circle_button_rect.collidepoint(pos):
                new_selected_shape = CIRCLE
                new_drawing = False
                new_adding_joint = False
                new_dragging_object = None
                new_dragged_complex = []
            elif weld_joint_button_rect.collidepoint(pos):
                new_adding_joint = not adding_joint if joint_type == WELD_JOINT else True
                new_joint_type = WELD_JOINT
                new_selected_shape = None
                new_drawing = False
                new_dragging_object = None
                new_dragged_complex = []
            elif pivot_joint_button_rect.collidepoint(pos):
                new_adding_joint = not adding_joint if joint_type == PIVOT_JOINT else True
                new_joint_type = PIVOT_JOINT
                new_selected_shape = None
                new_drawing = False
                new_dragging_object = None
                new_dragged_complex = []
            # --- Joint Creation Logic ---
            elif adding_joint:
                p_world = from_pygame(pos, camera_offset)
                search_radius = 15
                objects_at_pos = []
                # Find overlapping objects
                for obj in reversed(shapes):
                    # ... (precise overlap check logic - keep this) ...
                    try:
                        if isinstance(obj.shape, pymunk.Circle):
                            dist_sq = obj.body.position.get_dist_sqrd(p_world)
                            if dist_sq <= (obj.shape.radius + search_radius)**2:
                                objects_at_pos.append(obj)
                        elif isinstance(obj.shape, pymunk.Poly):
                             info = obj.shape.point_query(p_world)
                             if info.distance <= search_radius:
                                 objects_at_pos.append(obj)
                    except Exception as e:
                        print(f"Overlap check error: {e}")

                print(f"Found {len(objects_at_pos)} objects near {p_world} for {joint_type} joint.")
                if len(objects_at_pos) >= 2:
                    body1 = objects_at_pos[0].body
                    body2 = objects_at_pos[1].body
                    body1_id = id(body1)
                    body2_id = id(body2)
                    
                    # Check if these bodies should already not collide
                    already_connected = False
                    for component in connected_components:
                        if body1_id in component and body2_id in component:
                            already_connected = True
                            break
                    
                    if not already_connected:
                        add_joint(space, joints, new_joint_type, body1, body2, p_world, connected_components)
                        print(f"{new_joint_type.capitalize()} joint created between bodies near {p_world}")
                        new_adding_joint = False # Turn off after success
                    else:
                        print("Bodies already connected.")
                else:
                    print("Need at least two overlapping objects near the click point.")
            # --- Shape Drawing Start ---
            elif selected_shape and not drawing:
                 new_drawing = True
                 new_start_pos = pos
                 new_adding_joint = False
                 new_dragging_object = None
                 new_dragged_complex = []
            # --- Edit Mode Drag Start (Updated for complex objects) ---
            else:
                ui_buttons = [start_button_rect, rect_button_rect, circle_button_rect,
                              weld_joint_button_rect, pivot_joint_button_rect]
                clicked_ui = any(btn.collidepoint(pos) for btn in ui_buttons)
                if not clicked_ui and not adding_joint and not drawing:
                    p = from_pygame(pos, camera_offset)
                    clicked_object = None
                    # Find the topmost object clicked
                    for obj in reversed(shapes):
                        try:
                            info = obj.shape.point_query(p)
                            is_inside = info.distance < 0 if hasattr(info, 'distance') else False
                            if is_inside:
                                clicked_object = obj
                                break
                        except Exception as e:
                            print(f"Drag start query error: {e}")

                    if clicked_object:
                        # Build the complex of connected objects using connected_components
                        new_dragged_complex = []  # Reset the complex
                        clicked_body_id = id(clicked_object.body)
                        
                        # Find which component contains the clicked object
                        for component in connected_components:
                            if clicked_body_id in component:
                                # Add all objects that belong to this component
                                for shape in shapes:
                                    if id(shape.body) in component:
                                        new_dragged_complex.append(shape)
                                break
                        
                        # If not part of any component, just drag the clicked object
                        if not new_dragged_complex:
                            new_dragged_complex = [clicked_object]
                            
                        # Store reference to the primary dragged object and offset
                        offset = clicked_object.body.position - p
                        new_dragging_object = clicked_object
                        new_drag_offset = offset
                        print(f"Dragging complex with {len(new_dragged_complex)} objects")
                        new_selected_shape = None
                        new_drawing = False
                        new_adding_joint = False

        elif game_mode == SIMULATION_MODE:
            if start_button_rect.collidepoint(pos):
                new_game_mode = EDIT_MODE
                new_dragging_object = None # Stop dragging if any
            else: # Try dragging in sim mode
                p = from_pygame(pos, camera_offset)
                clicked_object_sim = None
                for obj in reversed(shapes):
                     # ... (precise shape query logic - keep this) ...
                     try:
                         info = obj.shape.point_query(p)
                         is_inside = info.distance < 0 if hasattr(info, 'distance') else False
                         if is_inside:
                             clicked_object_sim = obj
                             break
                     except Exception as e:
                         print(f"Sim drag query error: {e}")

                if clicked_object_sim:
                    offset = clicked_object_sim.body.position - p
                    clicked_object_sim.body.activate()
                    new_dragging_object = clicked_object_sim
                    new_drag_offset = offset

    elif event.type == pygame.MOUSEMOTION:
        if dragging_object:
            p = from_pygame(pygame.mouse.get_pos(), camera_offset)
            if game_mode == EDIT_MODE:
                # Use existing dragged_complex or rebuild it if needed
                if not new_dragged_complex:
                    dragged_body_id = id(dragging_object.body)
                    for component in connected_components:
                        if dragged_body_id in component:
                            # Rebuild dragged_complex
                            new_dragged_complex = [obj for obj in shapes if id(obj.body) in component]
                            break
                    
                    # If still empty, just use the main dragged object
                    if not new_dragged_complex:
                        new_dragged_complex = [dragging_object]
                
                # Calculate the position delta for the primary dragged object
                target_pos = p + new_drag_offset
                delta_pos = target_pos - dragging_object.body.position
                
                # Move all objects in the complex by the same delta
                for obj in new_dragged_complex:
                    obj.body.position += delta_pos
                    # Make sure to activate the body so it registers position changes
                    obj.body.activate()
            
            elif game_mode == SIMULATION_MODE:
                # Apply velocity to follow mouse (existing logic)
                target_pos = p + new_drag_offset
                # Simple velocity adjustment (might need tuning)
                vel_adj = (target_pos - dragging_object.body.position) * 10 # Adjust multiplier for responsiveness
                dragging_object.body.velocity = vel_adj
                dragging_object.body.activate()
        elif drawing and selected_shape:
             pass # Handled in main loop for end_pos update

    elif event.type == pygame.MOUSEBUTTONUP:
        pos = pygame.mouse.get_pos()
        if dragging_object:
            # Stop dragging
            if game_mode == EDIT_MODE:
                # Put all objects in the complex to sleep
                for obj in new_dragged_complex:
                    obj.body.sleep()
            
            new_dragging_object = None
            new_drag_offset = None
            new_dragged_complex = []  # Clear the complex when done dragging
        
        elif selected_shape and drawing:
            # ... (existing shape creation logic - keep this) ...
            end_pos_up = pos
            x1, y1 = from_pygame(start_pos, camera_offset)
            x2, y2 = from_pygame(end_pos_up, camera_offset)

            if selected_shape == RECTANGLE:
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                if width > 5 and height > 5:
                    x = min(x1, x2)
                    y = max(y1, y2) # Pymunk Y is inverted
                    body, shape = create_rectangle(x, y - height, width, height)
                    add_shape(space, shapes, RECTANGLE, body, shape, BLUE)
            elif selected_shape == CIRCLE:
                center_x, center_y = from_pygame(start_pos, camera_offset)
                radius = math.dist(start_pos, end_pos_up)
                if radius > 5:
                    body, shape = create_circle(center_x, center_y, radius)
                    add_shape(space, shapes, CIRCLE, body, shape, RED)

            new_selected_shape = None # Stop shape selection
            new_drawing = False
            new_start_pos = (0, 0)

    # Return updated state including camera_offset
    return new_game_mode, new_selected_shape, new_drawing, new_start_pos, \
           new_adding_joint, new_joint_type, \
           new_dragging_object, new_drag_offset, new_dragged_complex, \
           new_camera_offset

def draw_shape_being_created(screen, selected_shape, drawing, start_pos, end_pos, camera_offset=(0, 0)):
    """
    Draws the shape being created (before it's added). Adjusts for camera offset.
    """
    if selected_shape and drawing and end_pos:
        # Apply camera offset to drawing preview
        if selected_shape == RECTANGLE:
            x1, y1 = start_pos
            x2, y2 = end_pos
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            x = min(x1, x2)
            y = min(y1, y2)
            pygame.draw.rect(screen, BLUE, (x, y, width, height), 2)
        elif selected_shape == CIRCLE:
            x1, y1 = start_pos
            x2, y2 = end_pos
            radius = (abs(x2 - x1) + abs(y2 - y1)) / 4
            x = x1
            y = y1
            pygame.draw.circle(screen, RED, (x, y), int(radius), 2)

def draw_shapes(screen, shapes):
    """
    Draws all the shapes.
    """
    for obj in shapes:
        obj.draw(screen)

def draw_joints(screen, joints):
    """
    Draws all the joints
    """
    for joint in joints:
        joint.draw(screen)

def create_floor(space):
    """
    Creates a static floor with extended width to match our continuous ground.
    """
    # Create a much wider floor (same extension as in the draw_continuous_ground function)
    extension = 5000
    floor_width = SCREEN_WIDTH + (extension * 2)  # Extend in both directions
    
    # Static body for the floor
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = (0, 0)  # Position will be at the center of the segment
    
    # Create a wide segment shape
    # Position it at the center point with extension in both directions
    shape = pymunk.Segment(body, (-extension, 40), (SCREEN_WIDTH + extension, 40), 5.0)
    shape.friction = 1.0
    shape.collision_type = 1  # Same collision type as other objects
    
    # Add to space
    space.add(body, shape)
    
    return body, shape

def draw_properties_menu(screen, selected_object, font):
    """Draws the properties menu for the selected object."""
    if not selected_object:
        return
        
    # Menu background
    menu_width = 200
    menu_height = 400
    menu_x = 10
    menu_y = 150
    pygame.draw.rect(screen, GRAY, (menu_x, menu_y, menu_width, menu_height))
    pygame.draw.rect(screen, BLACK, (menu_x, menu_y, menu_width, menu_height), 2)
    
    # Title
    title_y = menu_y + 10
    title_text = font.render("Properties", True, BLACK)
    screen.blit(title_text, (menu_x + 10, title_y))
    
    # Object type
    type_y = title_y + 40  
    object_type = "Rectangle" if isinstance(selected_object.shape, pymunk.Poly) else "Circle"
    type_text = font.render(f"Type: {object_type}", True, BLACK)
    screen.blit(type_text, (menu_x + 10, type_y))
    
    # Current properties with +/- buttons
    density_y = type_y + 30
    density_text = font.render(f"Density: {selected_object.mass:.1f}", True, BLACK)
    screen.blit(density_text, (menu_x + 10, density_y))
    
    # Density +/- buttons
    density_minus_rect = pygame.Rect(menu_x + 160, density_y, 15, 15)
    density_plus_rect = pygame.Rect(menu_x + 180, density_y, 15, 15)
    pygame.draw.rect(screen, (200, 50, 50), density_minus_rect)  # Red for minus
    pygame.draw.rect(screen, (50, 200, 50), density_plus_rect)   # Green for plus
    minus_text = font.render("-", True, WHITE)
    plus_text = font.render("+", True, WHITE)
    screen.blit(minus_text, (density_minus_rect.x + 4, density_minus_rect.y - 2))
    screen.blit(plus_text, (density_plus_rect.x + 3, density_plus_rect.y - 2))
    
    friction_y = density_y + 30
    friction_text = font.render(f"Friction: {selected_object.friction:.2f}", True, BLACK)
    screen.blit(friction_text, (menu_x + 10, friction_y))
    
    # Friction +/- buttons
    friction_minus_rect = pygame.Rect(menu_x + 160, friction_y, 15, 15)
    friction_plus_rect = pygame.Rect(menu_x + 180, friction_y, 15, 15)
    pygame.draw.rect(screen, (200, 50, 50), friction_minus_rect)
    pygame.draw.rect(screen, (50, 200, 50), friction_plus_rect)
    screen.blit(minus_text, (friction_minus_rect.x + 4, friction_minus_rect.y - 2))
    screen.blit(plus_text, (friction_plus_rect.x + 3, friction_plus_rect.y - 2))
    
    # Color RGB values with +/- buttons
    color_title_y = friction_y + 30
    color_title = font.render("Color:", True, BLACK)
    screen.blit(color_title, (menu_x + 10, color_title_y))
    
    # R value
    r_y = color_title_y + 30
    r_text = font.render(f"R: {selected_object.color[0]}", True, BLACK)
    screen.blit(r_text, (menu_x + 10, r_y))
    
    # R +/- buttons
    r_minus_rect = pygame.Rect(menu_x + 160, r_y, 15, 15)
    r_plus_rect = pygame.Rect(menu_x + 180, r_y, 15, 15)
    pygame.draw.rect(screen, (200, 50, 50), r_minus_rect)
    pygame.draw.rect(screen, (50, 200, 50), r_plus_rect)
    screen.blit(minus_text, (r_minus_rect.x + 4, r_minus_rect.y - 2))
    screen.blit(plus_text, (r_plus_rect.x + 3, r_plus_rect.y - 2))
    
    # G value
    g_y = r_y + 30
    g_text = font.render(f"G: {selected_object.color[1]}", True, BLACK)
    screen.blit(g_text, (menu_x + 10, g_y))
    
    # G +/- buttons
    g_minus_rect = pygame.Rect(menu_x + 160, g_y, 15, 15)
    g_plus_rect = pygame.Rect(menu_x + 180, g_y, 15, 15)
    pygame.draw.rect(screen, (200, 50, 50), g_minus_rect)
    pygame.draw.rect(screen, (50, 200, 50), g_plus_rect)
    screen.blit(minus_text, (g_minus_rect.x + 4, g_minus_rect.y - 2))
    screen.blit(plus_text, (g_plus_rect.x + 3, g_plus_rect.y - 2))
    
    # B value
    b_y = g_y + 30
    b_text = font.render(f"B: {selected_object.color[2]}", True, BLACK)
    screen.blit(b_text, (menu_x + 10, b_y))
    
    # B +/- buttons
    b_minus_rect = pygame.Rect(menu_x + 160, b_y, 15, 15)
    b_plus_rect = pygame.Rect(menu_x + 180, b_y, 15, 15)
    pygame.draw.rect(screen, (200, 50, 50), b_minus_rect)
    pygame.draw.rect(screen, (50, 200, 50), b_plus_rect)
    screen.blit(minus_text, (b_minus_rect.x + 4, b_minus_rect.y - 2))
    screen.blit(plus_text, (b_plus_rect.x + 3, b_plus_rect.y - 2))
    
    # Delete button at the bottom
    delete_y = b_y + 60
    delete_button_rect = pygame.Rect(menu_x + 50, delete_y, 100, 30)
    pygame.draw.rect(screen, (255, 0, 0), delete_button_rect)  # Red button
    delete_text = font.render("Delete", True, WHITE)
    delete_text_rect = delete_text.get_rect(center=delete_button_rect.center)
    screen.blit(delete_text, delete_text_rect)
    
    # Store the button rectangles for click detection
    buttons = {
        "density_minus": density_minus_rect,
        "density_plus": density_plus_rect,
        "friction_minus": friction_minus_rect,
        "friction_plus": friction_plus_rect,
        "r_minus": r_minus_rect,
        "r_plus": r_plus_rect,
        "g_minus": g_minus_rect,
        "g_plus": g_plus_rect,
        "b_minus": b_minus_rect,
        "b_plus": b_plus_rect,
        "delete": delete_button_rect
    }
    return buttons

def draw_joint_menu(screen, selected_joint, font):
    """Draws a simple menu for the selected joint with just a delete button."""
    if not selected_joint:
        return
        
    # Menu background
    menu_width = 200
    menu_height = 120
    menu_x = 10
    menu_y = 150
    pygame.draw.rect(screen, GRAY, (menu_x, menu_y, menu_width, menu_height))
    pygame.draw.rect(screen, BLACK, (menu_x, menu_y, menu_width, menu_height), 2)
    
    # Title
    title_y = menu_y + 10
    title_text = font.render("Joint Properties", True, BLACK)
    screen.blit(title_text, (menu_x + 10, title_y))
    
    # Joint type
    type_y = title_y + 40
    joint_type_text = font.render(f"Type: {selected_joint.joint_type}", True, BLACK)
    screen.blit(joint_type_text, (menu_x + 10, type_y))
    
    # Delete button
    delete_y = type_y + 40
    delete_button_rect = pygame.Rect(menu_x + 50, delete_y, 100, 30)
    pygame.draw.rect(screen, (255, 0, 0), delete_button_rect)  # Red button
    delete_text = font.render("Delete", True, WHITE)
    delete_text_rect = delete_text.get_rect(center=delete_button_rect.center)
    screen.blit(delete_text, delete_text_rect)
    
    # Return the delete button for click detection
    return {"delete": delete_button_rect}

def delete_shape(space, shapes, shape_obj):
    """Removes a shape from the space and the shapes list."""
    try:
        # Remove from pymunk space
        space.remove(shape_obj.body, shape_obj.shape)
        # Remove from our list
        shapes.remove(shape_obj)
        print(f"Deleted shape: {type(shape_obj).__name__}")
        return True
    except Exception as e:
        print(f"Error deleting shape: {e}")
        return False

def delete_joint(space, joints, joint_obj, connected_components):
    """Removes a joint from the space and the joints list."""
    try:
        # Wake up the connected bodies if they're sleeping
        if joint_obj.body_a and joint_obj.body_a.is_sleeping:
            joint_obj.body_a.activate()
        if joint_obj.body_b and joint_obj.body_b.is_sleeping:
            joint_obj.body_b.activate()
            
        # For a weld joint, we need to remove both the pivot and gear joints
        if joint_obj.joint_type == WELD_JOINT:
            # Remove the gear joint if we stored it
            if hasattr(joint_obj, 'gear_joint'):
                space.remove(joint_obj.gear_joint)
            else:
                # Fallback to finding the gear joint if not stored
                for constraint in space.constraints:
                    if isinstance(constraint, pymunk.GearJoint) and \
                       constraint.a == joint_obj.body_a and constraint.b == joint_obj.body_b:
                        space.remove(constraint)
                        break
        # For a pivot joint with a slide constraint, remove that as well
        elif joint_obj.joint_type == PIVOT_JOINT and hasattr(joint_obj, 'slide_joint'):
            space.remove(joint_obj.slide_joint)
        
        # Remove the main joint from pymunk
        space.remove(joint_obj.joint)
        
        # Remove from our list
        joints.remove(joint_obj)
        
        # After removing a joint, we need to recalculate all connected components
        # This is more complex than the add case but ensures correctness
        body_a_id = id(joint_obj.body_a) if joint_obj.body_a else None
        body_b_id = id(joint_obj.body_b) if joint_obj.body_b else None
        
        if body_a_id and body_b_id:
            # We need to rebuild connected components from remaining joints
            # First, clear all current components
            connected_components.clear()
            
            # For each remaining joint, add its bodies to components
            for j in joints:
                if j.body_a and j.body_b:
                    j_body_a_id = id(j.body_a)
                    j_body_b_id = id(j.body_b)
                    
                    # Find which component each body belongs to
                    comp_a = None
                    comp_b = None
                    comp_a_idx = -1
                    comp_b_idx = -1
                    
                    for i, component in enumerate(connected_components):
                        if j_body_a_id in component:
                            comp_a = component
                            comp_a_idx = i
                        if j_body_b_id in component:
                            comp_b = component
                            comp_b_idx = i
                    
                    # Apply same logic as add_joint
                    if comp_a is not None and comp_a is comp_b:
                        pass  # Nothing to do
                    elif comp_a is not None and comp_b is None:
                        connected_components[comp_a_idx].add(j_body_b_id)
                    elif comp_a is None and comp_b is not None:
                        connected_components[comp_b_idx].add(j_body_a_id)
                    elif comp_a is not None and comp_b is not None:
                        merged_component = comp_a.union(comp_b)
                        connected_components[comp_a_idx] = merged_component
                        connected_components.pop(comp_b_idx)
                    else:
                        connected_components.append({j_body_a_id, j_body_b_id})
        
        print(f"Deleted joint: {joint_obj.joint_type}")
        return True
    except Exception as e:
        print(f"Error deleting joint: {e}")
        return False

def handle_property_change(obj, property_name, increase=True):
    """Handles changing an object's properties."""
    try:
        # Make sure object is active before modifying
        if obj.body.is_sleeping:
            obj.body.activate()
            
        # Determine amount to change
        if property_name == "density":
            delta = 1.0 if increase else -1.0
            obj.mass = max(1.0, obj.mass + delta)
        elif property_name == "friction":
            delta = 0.05 if increase else -0.05
            obj.friction = max(0.0, min(1.0, obj.friction + delta))
            obj.shape.friction = obj.friction
        elif property_name == "color_r":
            delta = 10 if increase else -10
            r = max(0, min(255, obj.color[0] + delta))
            obj.color = (r, obj.color[1], obj.color[2])
        elif property_name == "color_g":
            delta = 10 if increase else -10
            g = max(0, min(255, obj.color[1] + delta))
            obj.color = (obj.color[0], g, obj.color[2])
        elif property_name == "color_b":
            delta = 10 if increase else -10
            b = max(0, min(255, obj.color[2] + delta))
            obj.color = (obj.color[0], obj.color[1], b)
    except Exception as e:
        print(f"Error changing property {property_name}: {e}")

def safe_body_operation(body, operation, *args):
    """Safely performs an operation on a body, handling potential errors."""
    try:
        if body and not body._removed:
            return operation(*args)
    except Exception as e:
        print(f"Body operation error: {e}")
    return None

def draw_continuous_ground(screen, camera_offset):
    """
    Draws a continuous ground that extends beyond the visible screen.
    """
    # Calculate how far the ground should extend in each direction beyond the screen
    extension = 5000  # A large value to ensure the ground appears continuous
    
    # Calculate ground position in screen coordinates
    ground_y = to_pygame((0, 40), camera_offset)[1]  # Y-coordinate of ground in screen space
    
    # Draw the extended ground line
    ground_left = to_pygame((-extension, 40), camera_offset)[0]
    ground_right = to_pygame((SCREEN_WIDTH + extension, 40), camera_offset)[0]
    
    pygame.draw.line(screen, BLACK, (ground_left, ground_y), (ground_right, ground_y), 10)
    
    # Optional: Add visual indicators every 100 pixels to show movement
    for x in range(-extension, SCREEN_WIDTH + extension, 100):
        tick_x = to_pygame((x, 40), camera_offset)[0]
        pygame.draw.line(screen, (80, 80, 80), (tick_x, ground_y - 5), (tick_x, ground_y + 5), 2)

def run_simulation():
    """
    Runs the main game loop.
    """
    global font, start_button_rect, rect_button_rect, circle_button_rect
    global weld_joint_button_rect, pivot_joint_button_rect
    
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("2D Physics Simulator")
    
    # Initialize fonts early
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 20)
    
    space = pymunk.Space()
    space.gravity = GRAVITY
    space.sleep_time_threshold = 0.5
    connected_components = []  # Change from connected_bodies to connected_components
    handler = space.add_collision_handler(1, 1)
    handler.begin = lambda arb, sp, data: should_collide(arb.shapes[0].body, arb.shapes[1].body, connected_components)
    floor_body, floor_shape = create_floor(space)

    # Game state
    game_mode = EDIT_MODE
    selected_shape = None
    drawing = False
    start_pos = (0, 0)
    end_pos = None # For drawing preview
    shapes = []
    joints = []
    adding_joint = False
    joint_type = WELD_JOINT # Default joint type
    initial_body_states = {}
    dragging_object = None
    drag_offset = None
    dragged_complex = [] # This will now persist between events
    selected_shape_object = None # For properties menu
    selected_joint_object = None # For joint menu

    # Fonts
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 20) # Smaller font for buttons

    # --- UI Elements ---
    global start_button_rect, rect_button_rect, circle_button_rect
    global weld_joint_button_rect, pivot_joint_button_rect # Updated joint buttons

    start_button_rect = pygame.Rect(SCREEN_WIDTH - 110, 5, 100, 30) # Top row
    start_button_text = font.render("Start", True, WHITE)
    stop_button_text = font.render("Stop", True, WHITE)

    rect_button_rect = pygame.Rect(10, 5, 110, 30) # Top row
    circle_button_rect = pygame.Rect(130, 5, 80, 30) # Top row
    rect_button_text = font.render("Rectangle", True, WHITE)
    circle_button_text = font.render("Circle", True, WHITE)

    # Joint buttons on second row
    weld_joint_button_rect = pygame.Rect(10, 45, 90, 25)
    weld_joint_button_text = small_font.render("Add Weld", True, WHITE) # White text
    pivot_joint_button_rect = pygame.Rect(110, 45, 130, 25) # Next to weld
    pivot_joint_button_text = small_font.render("Add Rotating Joint", True, WHITE) # White text

    clock = pygame.time.Clock()
    running = True
    property_buttons = {} # For properties menu
    joint_menu_buttons = {} # For joint menu

    # Initialize camera position
    camera_offset = (0, 0)
    camera_speed = CAMERA_MOVE_SPEED

    while running:
        # Store previous mode for state change detection
        previous_mode = game_mode

        # --- Custom Cursor ---
        show_custom_cursor = adding_joint and joint_type == PIVOT_JOINT and game_mode == EDIT_MODE
        pygame.mouse.set_visible(not show_custom_cursor)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break # Exit event loop immediately

            # Initialize variable for property button clicks
            clicked_prop_button = False
            
            # Handle property button clicks first if menu is active
            if game_mode == EDIT_MODE and event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                
                # Handle object property buttons
                if selected_shape_object:
                    for btn_name, btn_rect in property_buttons.items():
                        if btn_rect.collidepoint(pos):
                            if btn_name == "delete":
                                # Delete the selected shape
                                if delete_shape(space, shapes, selected_shape_object):
                                    selected_shape_object = None
                            else:
                                # Extract property name from button name
                                property_name = None
                                if "density" in btn_name:
                                    property_name = "density"
                                elif "friction" in btn_name:
                                    property_name = "friction"  
                                elif "r_" in btn_name:
                                    property_name = "color_r"
                                elif "g_" in btn_name:
                                    property_name = "color_g"
                                elif "b_" in btn_name:
                                    property_name = "color_b"
                                
                                if property_name:
                                    handle_property_change(selected_shape_object, 
                                                         property_name, 
                                                         "plus" in btn_name)
                            clicked_prop_button = True
                            break
                
                # Handle joint menu buttons
                if selected_joint_object and not clicked_prop_button:
                    for btn_name, btn_rect in joint_menu_buttons.items():
                        if btn_rect.collidepoint(pos):
                            if btn_name == "delete":
                                # Delete the selected joint
                                if delete_joint(space, joints, selected_joint_object, connected_components):
                                    selected_joint_object = None
                            clicked_prop_button = True
                            break
                    
                if clicked_prop_button:
                    continue  # Skip the rest of event processing for this click

            # Handle selection click (only if not clicking properties)
            if game_mode == EDIT_MODE and event.type == pygame.MOUSEBUTTONDOWN and \
               not clicked_prop_button and not drawing and not adding_joint:
                pos = pygame.mouse.get_pos()
                ui_buttons = [start_button_rect, rect_button_rect, circle_button_rect,
                             weld_joint_button_rect, pivot_joint_button_rect]
                clicked_ui = any(btn.collidepoint(pos) for btn in ui_buttons)

                if not clicked_ui:
                    # Clear current selections
                    selected_shape_object = None
                    selected_joint_object = None
                    
                    # Check for joint selection first
                    p = from_pygame(pos, camera_offset)
                    clicked_joint = None
                    joint_selection_distance = 15  # Radius to check for joint selection
                    
                    for joint in joints:
                        # For PinJoint or PivotJoint, check if click is near joint point
                        if isinstance(joint, PivotJoint) or isinstance(joint, WeldJoint):
                            if joint.body_a:  # Ensure body_a exists
                                try:
                                    # Use safe operation to get joint position
                                    def get_joint_pos(anchor_local):
                                        return joint.body_a.local_to_world(anchor_local)
                                    
                                    joint_pos = safe_body_operation(joint.body_a, 
                                                                  get_joint_pos, 
                                                                  joint.anchor_a_local)
                                    
                                    dist = math.sqrt((joint_pos[0] - p[0])**2 + (joint_pos[1] - p[1])**2)
                                    if dist <= joint_selection_distance:
                                        clicked_joint = joint
                                        break
                                except Exception as e:
                                    print(f"Joint selection error: {e}")
                                    continue
                    
                    if clicked_joint:
                        selected_joint_object = clicked_joint
                        print(f"Selected joint: {clicked_joint.joint_type}")
                    else:
                        # Check for shape selection
                        for obj in reversed(shapes):
                            try:
                                info = obj.shape.point_query(p)
                                is_inside = info.distance < 0 if hasattr(info, 'distance') else False
                                if is_inside:
                                    selected_shape_object = obj
                                    print(f"Selected object: {type(selected_shape_object)}")
                                    break
                            except Exception as e:
                                print(f"Selection query error: {e}")

            # Pass camera_offset to handle_input but don't handle arrow keys there
            game_mode, selected_shape, drawing, start_pos, adding_joint, joint_type, \
            dragging_object, drag_offset, dragged_complex, camera_offset = handle_input(
                event, game_mode, selected_shape, drawing, start_pos, shapes, space,
                adding_joint, joint_type, joints, connected_components,
                dragging_object, drag_offset, dragged_complex,
                camera_offset
            )

            # Update end_pos for drawing preview
            if event.type == pygame.MOUSEMOTION and drawing:
                 end_pos = pygame.mouse.get_pos()
            elif event.type != pygame.MOUSEMOTION:
                 end_pos = None

        # Handle camera movement with key press and hold
        # This works in both edit and simulation modes
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            camera_offset = (camera_offset[0] - camera_speed, camera_offset[1])
        if keys[pygame.K_RIGHT]:
            camera_offset = (camera_offset[0] + camera_speed, camera_offset[1])
        if keys[pygame.K_UP]:
            camera_offset = (camera_offset[0], camera_offset[1] - camera_speed)
        if keys[pygame.K_DOWN]:
            camera_offset = (camera_offset[0], camera_offset[1] + camera_speed)
        
        # --- State Management for Simulation Start/Stop ---
        mode_changed = previous_mode != game_mode
        if mode_changed:
            if game_mode == SIMULATION_MODE:
                print("Starting Simulation")
                initial_body_states = {id(obj.body): (obj.body.position, obj.body.angle, obj.body.velocity, obj.body.angular_velocity) for obj in shapes}
                for obj in shapes:
                    obj.apply_mass()
                    obj.body.activate()
                selected_shape_object = None
                selected_joint_object = None
                adding_joint = False
            elif game_mode == EDIT_MODE:
                print("Stopping Simulation - Restoring Edit State")
                for obj in shapes:
                    body_id = id(obj.body)
                    if body_id in initial_body_states:
                        pos, angle, vel, ang_vel = initial_body_states[body_id]
                        obj.body.position = pos
                        obj.body.angle = angle
                        obj.body.velocity = (0, 0)
                        obj.body.angular_velocity = 0
                    obj.body.sleep()
                dragging_object = None
                drag_offset = None

        # --- Drawing (with camera offset) ---
        screen.fill(WHITE)
        
        # Draw continuous ground
        draw_continuous_ground(screen, camera_offset)
        
        # Draw all shapes with camera offset
        for obj in shapes:
            # Override the draw methods to account for camera
            if isinstance(obj, Rectangle):
                # Get vertices relative to the body's center of gravity
                local_verts = obj.shape.get_vertices()
                # Transform local vertices to world coordinates 
                world_verts = [obj.body.local_to_world(v) for v in local_verts]
                # Convert world coordinates to Pygame screen coordinates with camera offset
                pygame_verts = [to_pygame(v, camera_offset) for v in world_verts]
                # Draw the polygon
                pygame.draw.polygon(screen, obj.color, pygame_verts)
            elif isinstance(obj, Circle):
                position = to_pygame(obj.body.position, camera_offset)
                pygame.draw.circle(screen, obj.color, position, int(obj.radius))
                angle_radians = obj.body.angle
                end_point_x = position[0] + obj.radius * math.cos(angle_radians)
                end_point_y = position[1] + obj.radius * math.sin(angle_radians)
                pygame.draw.line(screen, BLACK, position, (end_point_x, end_point_y), 2)
        
        # Draw joints with camera offset
        for joint in joints:
            if isinstance(joint, PinJoint):
                if joint.body_a:
                    pos_a = to_pygame(joint.body_a.position + joint.anchor_a, camera_offset)
                else:
                    pos_a = to_pygame(joint.anchor_a, camera_offset)
                if joint.body_b:
                    pos_b = to_pygame(joint.body_b.position + joint.anchor_b, camera_offset)
                else:
                    pos_b = to_pygame(joint.anchor_b, camera_offset)
                pygame.draw.line(screen, GREEN, pos_a, pos_b, 2)
                pygame.draw.circle(screen, RED, pos_a, 5)
                pygame.draw.circle(screen, RED, pos_b, 5)
            elif isinstance(joint, PivotJoint):
                if joint.body_a:
                    current_joint_world_pos = joint.body_a.local_to_world(joint.anchor_a_local)
                    joint_pos_pygame = to_pygame(current_joint_world_pos, camera_offset)
                    pygame.draw.circle(screen, RED, joint_pos_pygame, 6)
                    pygame.draw.circle(screen, BLACK, joint_pos_pygame, 6, 1)
                    
                    if joint.body_b:
                        pos_a = to_pygame(joint.body_a.position, camera_offset)
                        pos_b = to_pygame(joint.body_b.position, camera_offset)
                        pygame.draw.line(screen, GRAY, joint_pos_pygame, pos_a, 1)
                        pygame.draw.line(screen, GRAY, joint_pos_pygame, pos_b, 1)
            elif isinstance(joint, WeldJoint):
                if joint.body_a:
                    current_joint_world_pos = joint.body_a.local_to_world(joint.anchor_a_local)
                    joint_pos_pygame = to_pygame(current_joint_world_pos, camera_offset)
                    
                    square_size = 10
                    pygame.draw.rect(screen, BLACK,
                                    (joint_pos_pygame[0] - square_size//2,
                                     joint_pos_pygame[1] - square_size//2,
                                     square_size, square_size))
                    
                    if joint.body_b:
                        pos_a = to_pygame(joint.body_a.position, camera_offset)
                        pos_b = to_pygame(joint.body_b.position, camera_offset)
                        pygame.draw.line(screen, BLACK, joint_pos_pygame, pos_a, 1)
                        pygame.draw.line(screen, BLACK, joint_pos_pygame, pos_b, 1)
        
        # Draw shape being created with camera offset
        if drawing and start_pos and end_pos:
             draw_shape_being_created(screen, selected_shape, drawing, start_pos, end_pos)
        
        # Draw UI elements (no camera offset since they're in screen space)
        draw_ui(screen, game_mode, start_button_rect, start_button_text, stop_button_text,
                weld_joint_button_rect, weld_joint_button_text,
                pivot_joint_button_rect, pivot_joint_button_text,
                rect_button_rect, rect_button_text,
                circle_button_rect, circle_button_text,
                adding_joint, joint_type, camera_offset)
        
        # Draw Properties Menu if an object is selected
        if game_mode == EDIT_MODE:
            if selected_shape_object:
                property_buttons = draw_properties_menu(screen, selected_shape_object, font) or {}
                selected_joint_object = None  # Clear joint selection when shape is selected
            elif selected_joint_object:
                joint_menu_buttons = draw_joint_menu(screen, selected_joint_object, font) or {}
                selected_shape_object = None  # Clear shape selection when joint is selected

        # --- Draw Custom Cursor ---
        if show_custom_cursor:
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.circle(screen, RED, mouse_pos, 8, 2) # Red circle cursor

        # Physics simulation update
        if game_mode == SIMULATION_MODE:
            dt = 1.0 / FPS
            space.step(dt)
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_simulation()