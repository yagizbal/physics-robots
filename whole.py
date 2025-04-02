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
SKY_BLUE = (135, 206, 235)
LIGHT_ORANGE = (245, 155, 66)
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

    def draw(self, screen, camera_offset=(0, 0)):
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

    def draw(self, screen, camera_offset=(0, 0)):
        """Draws the rectangle on the screen."""
        local_verts = self.shape.get_vertices()
        world_verts = [self.body.local_to_world(v) for v in local_verts]
        pygame_verts = [to_pygame(v, camera_offset) for v in world_verts]
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

    def draw(self, screen, camera_offset=(0, 0)):
        """Draws the circle on the screen."""
        position = to_pygame(self.body.position, camera_offset)
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

    def draw(self, screen, camera_offset=(0, 0)):
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

    def draw(self, screen, camera_offset=(0, 0)):
        """Draws the pin joint."""
        if self.body_a:
            pos_a = to_pygame(self.body_a.position + self.anchor_a, camera_offset)
        else:
            pos_a = to_pygame(self.anchor_a, camera_offset)
        if self.body_b:
            pos_b = to_pygame(self.body_b.position + self.anchor_b, camera_offset)
        else:
            pos_b = to_pygame(self.anchor_b, camera_offset)
        pygame.draw.line(screen, BLACK, pos_a, pos_b, 2)
        pygame.draw.circle(screen, BLACK, pos_a, 5)
        pygame.draw.circle(screen, BLACK, pos_b, 5)

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

    def draw(self, screen, camera_offset=(0, 0)):
        """Draws the pivot joint."""
        if not self.body_a:
            return

        current_joint_world_pos = self.body_a.local_to_world(self.anchor_a_local)
        joint_pos_pygame = to_pygame(current_joint_world_pos, camera_offset)

        pygame.draw.circle(screen, BLACK, joint_pos_pygame, 6)
        pygame.draw.circle(screen, BLACK, joint_pos_pygame, 6, 1)

        if self.body_b:
            pos_a = to_pygame(self.body_a.position, camera_offset)
            pos_b = to_pygame(self.body_b.position, camera_offset)
            pygame.draw.line(screen, BLACK, joint_pos_pygame, pos_a, 1)
            pygame.draw.line(screen, BLACK, joint_pos_pygame, pos_b, 1)

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

    def draw(self, screen, camera_offset=(0, 0)):
        """Draws the weld joint."""
        if not self.body_a:
             return

        current_joint_world_pos = self.body_a.local_to_world(self.anchor_a_local)
        joint_pos_pygame = to_pygame(current_joint_world_pos, camera_offset)

        square_size = 10
        pygame.draw.rect(screen, BLACK,
                        (joint_pos_pygame[0] - square_size//2,
                         joint_pos_pygame[1] - square_size//2,
                         square_size, square_size))

        if self.body_b:
            pos_a = to_pygame(self.body_a.position, camera_offset)
            pos_b = to_pygame(self.body_b.position, camera_offset)
            pygame.draw.line(screen, BLACK, joint_pos_pygame, pos_a, 1)
            pygame.draw.line(screen, BLACK, joint_pos_pygame, pos_b, 1)

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
    Only affects rendering - converts world coordinates to screen coordinates.
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
    Only affects input - converts screen coordinates to world coordinates.
    """
    return p[0] + camera_offset[0], SCREEN_HEIGHT - p[1] + camera_offset[1]

def add_shape(space, shapes, shape_type, body, shape, color=LIGHT_ORANGE):
    """
    Adds a shape to the space and the list of shapes.
    """
    if shape_type == RECTANGLE:
        obj = Rectangle(body, shape, color, shape.bb.right - shape.bb.left, shape.bb.top - shape.bb.bottom)
    elif shape_type == CIRCLE:
        obj = Circle(body, shape, color, shape.radius)
    shapes.append(obj)
    space.add(body, shape)

    # Make sure the body is awake
    body.activate()
    
    # Return the newly created object (useful for selection)
    return obj

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
        # Only apply camera offset for world interactions, not UI elements
        if game_mode == EDIT_MODE:
            # --- Button Click Handling --- (UI elements - no camera offset)
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
            # --- Joint Creation Logic --- (World interaction - apply camera offset)
            elif adding_joint:
                # Convert screen coordinates to world coordinates with camera offset
                p_world = from_pygame(pos, camera_offset)
                print(f"Joint placement attempt at screen coords {pos}, world coords {p_world} with camera offset {camera_offset}")
                search_radius = 15 # How close the click needs to be
                objects_near_click = [] # Store tuples of (distance_sq, object)

                # Find overlapping objects and calculate distance to click point
                for obj in shapes: # Iterate through all shapes
                    try:
                        # Use body position as the reference point for distance calculation
                        obj_pos = obj.body.position
                        dist_sq = obj_pos.get_dist_sqrd(p_world)
                        is_close_enough = False

                        # Check if click is within search radius of the object's bounds
                        if isinstance(obj.shape, pymunk.Circle):
                            # Check distance from center + radius against search radius
                            if dist_sq <= (obj.shape.radius + search_radius)**2:
                                is_close_enough = True
                        elif isinstance(obj.shape, pymunk.Poly):
                            # For polys, check if the click is near the bounding box first for efficiency
                            # A simple check: distance from center vs rough diagonal + search radius
                            # This is an approximation but avoids complex point-in-poly checks here
                            half_width = (obj.shape.bb.right - obj.shape.bb.left) / 2
                            half_height = (obj.shape.bb.top - obj.shape.bb.bottom) / 2
                            rough_radius_sq = half_width**2 + half_height**2
                            if dist_sq <= rough_radius_sq + search_radius**2 * 2: # Heuristic check
                                # More precise check (optional but better):
                                info = obj.shape.point_query(p_world)
                                if info.distance <= search_radius:
                                     is_close_enough = True

                        if is_close_enough:
                            objects_near_click.append((dist_sq, obj))
                            print(f"Found object {type(obj).__name__} at {obj_pos}, dist_sq: {dist_sq}")

                    except Exception as e:
                        print(f"Overlap/distance check error: {e}")

                # Sort the found objects by distance (closest first)
                objects_near_click.sort(key=lambda item: item[0])

                print(f"Found {len(objects_near_click)} objects near click, sorted by distance.")

                if len(objects_near_click) >= 2:
                    # Select the two closest objects
                    body1 = objects_near_click[0][1].body
                    body2 = objects_near_click[1][1].body
                    body1_id = id(body1)
                    body2_id = id(body2)

                    print(f"Selected closest bodies: {type(objects_near_click[0][1]).__name__} and {type(objects_near_click[1][1]).__name__}")

                    # Check if these bodies are already connected
                    already_connected = False
                    for component in connected_components:
                        if body1_id in component and body2_id in component:
                            already_connected = True
                            print("Selected bodies are already connected.")
                            break

                    if not already_connected:
                        # Add the joint using the world click position as the anchor
                        add_joint(space, joints, new_joint_type, body1, body2, p_world, connected_components)
                        print(f"{new_joint_type.capitalize()} joint created between bodies near {p_world}")
                        new_adding_joint = False # Turn off after successful joint creation
                    # else: # Already printed message above
                    #    pass
                else:
                    print("Need at least two objects near the click point to create a joint.")
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
                    try:
                        p = from_pygame(pos, camera_offset)
                        clicked_object = None
                        # Find the topmost object clicked
                        for obj in reversed(shapes):
                            try:
                                # Wake up the body first to prevent sleeping-related errors
                                if obj.body.is_sleeping:
                                    # print(f"Waking up body for drag operation") # Less verbose
                                    obj.body.activate()

                                # Simple distance-based check for circles
                                if isinstance(obj.shape, pymunk.Circle):
                                    dist = math.sqrt((obj.body.position.x - p[0])**2 +
                                                     (obj.body.position.y - p[1])**2)
                                    if dist <= obj.shape.radius:
                                        clicked_object = obj
                                        break
                                # Polygons
                                elif isinstance(obj.shape, pymunk.Poly):
                                    vertices = [obj.body.local_to_world(v) for v in obj.shape.get_vertices()]
                                    if point_in_polygon(p, vertices):
                                        clicked_object = obj
                                        break
                            except Exception as e:
                                print(f"Drag selection error: {e}")

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
                    except Exception as e:
                        print(f"Error starting drag: {e}")
                        # Don't crash on drag errors

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
                    # Check if body still exists before trying to sleep
                    if obj.body and not obj.body._removed:
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
                    # Use the add_shape function and store the returned object
                    newly_added_object = add_shape(space, shapes, RECTANGLE, body, shape)
                    # Automatically select the new object
                    selected_shape_object = newly_added_object
            elif selected_shape == CIRCLE:
                center_x, center_y = from_pygame(start_pos, camera_offset)
                radius = math.dist(start_pos, end_pos_up)
                if radius > 5:
                    body, shape = create_circle(center_x, center_y, radius)
                    # Use the add_shape function and store the returned object
                    newly_added_object = add_shape(space, shapes, CIRCLE, body, shape)
                    # Automatically select the new object
                    selected_shape_object = newly_added_object

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
    Draws in screen coordinates relative to the camera view.
    """
    if selected_shape and drawing and end_pos:
        # start_pos and end_pos are already screen coordinates relative to the window.
        # No camera offset needed here as we are drawing a temporary overlay in screen space.
        if selected_shape == RECTANGLE:
            x1, y1 = start_pos
            x2, y2 = end_pos
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            x = min(x1, x2)
            y = min(y1, y2)
            pygame.draw.rect(screen, LIGHT_ORANGE, (x, y, width, height), 2) # Draw outline
        elif selected_shape == CIRCLE:
            x1, y1 = start_pos
            radius = math.dist(start_pos, end_pos) # Use actual distance for radius
            if radius > 0:
                 pygame.draw.circle(screen, LIGHT_ORANGE, (x1, y1), int(radius), 2) # Draw outline centered at start

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
    
    pygame.draw.line(screen, GREEN, (ground_left, ground_y), (ground_right, ground_y), 10)
    
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
    initial_body_states = {}  # Store initial positions of all bodies
    original_states = {}      # Persistent storage of original positions
    dragging_object = None
    drag_offset = None
    dragged_complex = [] # This will now persist between events
    selected_shape_object = None # For properties menu
    selected_joint_object = None # For joint menu

    # Initialize camera position
    camera_offset = (0, 0)
    camera_speed = CAMERA_MOVE_SPEED

    # Track cursor in both screen and world coordinates
    screen_cursor_pos = (0, 0)
    world_cursor_pos = (0, 0)

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

    # Transparency factor for selected object
    selected_alpha_blend = 0.6 # 60% object color, 40% background color

    # For tracking the most recently added object
    newly_added_object = None

    while running:
        # Store previous mode for state change detection
        previous_mode = game_mode

        # --- Update cursor positions ---
        screen_cursor_pos = pygame.mouse.get_pos()
        world_cursor_pos = from_pygame(screen_cursor_pos, camera_offset)

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
                pos = screen_cursor_pos  # Use tracked screen cursor position

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
                pos = screen_cursor_pos  # Use tracked screen cursor position
                ui_buttons = [start_button_rect, rect_button_rect, circle_button_rect,
                             weld_joint_button_rect, pivot_joint_button_rect]
                clicked_ui = any(btn.collidepoint(pos) for btn in ui_buttons)

                if not clicked_ui:
                    print("Processing selection click...")
                    try:
                        # Get world coordinates for the click - now using our tracked world cursor position
                        p = world_cursor_pos
                        print(f"Click at world coordinates: {p}")

                        # Track previous selection to handle re-selection properly
                        prev_shape = selected_shape_object
                        prev_joint = selected_joint_object

                        # Always clear selection first - this is critical
                        print("Clearing previous selection")
                        selected_shape_object = None
                        selected_joint_object = None
                        property_buttons = {}
                        joint_menu_buttons = {}

                        # Try to select a joint first
                        joint_selection_distance = 15
                        print(f"Checking {len(joints)} joints for selection...")
                        for joint in joints:
                            try:
                                if isinstance(joint, PivotJoint) or isinstance(joint, WeldJoint):
                                    if joint.body_a:
                                        # Wake up the body to avoid sleeping issues
                                        if joint.body_a.is_sleeping:
                                            print(f"Waking up joint body_a")
                                            joint.body_a.activate()

                                        # Calculate joint position
                                        try:
                                            joint_pos = joint.body_a.local_to_world(joint.anchor_a_local)
                                            dist = math.sqrt((joint_pos[0] - p[0])**2 + (joint_pos[1] - p[1])**2)

                                            if dist <= joint_selection_distance:
                                                print(f"Selected joint: {joint.joint_type} at {joint_pos}")
                                                selected_joint_object = joint
                                                # Wake up both bodies
                                                if joint.body_b and joint.body_b.is_sleeping:
                                                    joint.body_b.activate()
                                                break
                                        except Exception as e:
                                            print(f"Error calculating joint position: {e}")
                            except Exception as e:
                                print(f"Joint selection error: {e}")

                        # If no joint was selected, try to select a shape
                        if not selected_joint_object:
                            print(f"No joint selected. Checking {len(shapes)} shapes...")
                            for obj in reversed(shapes):
                                try:
                                    # Wake up the body first to prevent sleeping-related errors
                                    if obj.body.is_sleeping:
                                        print(f"Waking up shape body")
                                        obj.body.activate()

                                    # Simple distance-based selection for circles
                                    if isinstance(obj.shape, pymunk.Circle):
                                        dist = math.sqrt((obj.body.position.x - p[0])**2 +
                                                        (obj.body.position.y - p[1])**2)
                                        if dist <= obj.shape.radius:
                                            print(f"Selected circle at {obj.body.position} with radius {obj.shape.radius}")
                                            selected_shape_object = obj
                                            break
                                    # Polygon/rectangle selection
                                    elif isinstance(obj.shape, pymunk.Poly):
                                        try:
                                            # Use safer boundary check instead of point_query
                                            vertices = [obj.body.local_to_world(v) for v in obj.shape.get_vertices()]
                                            if point_in_polygon(p, vertices):
                                                print(f"Selected polygon with {len(vertices)} vertices")
                                                selected_shape_object = obj
                                                break
                                        except Exception as e:
                                            print(f"Error in polygon selection: {e}")
                                except Exception as e:
                                    print(f"Shape selection error: {e}")

                        if not selected_shape_object and not selected_joint_object:
                            print("Selection cleared (clicked background).")
                        elif prev_shape != selected_shape_object or prev_joint != selected_joint_object:
                            print("Selection changed.")

                    except Exception as e:
                        print(f"*** CRITICAL: Selection error: {e}")
                        # Don't crash the whole app on selection errors
                        import traceback
                        traceback.print_exc()

            # Handle all other input, now using tracked cursor positions
            if event.type == pygame.MOUSEBUTTONDOWN:
                if game_mode == EDIT_MODE:
                    # Button click handling - use screen coordinates
                    pos = screen_cursor_pos
                    if start_button_rect.collidepoint(pos):
                        game_mode = SIMULATION_MODE
                        selected_shape = None
                        drawing = False
                        adding_joint = False
                        dragging_object = None
                        dragged_complex = []
                    elif rect_button_rect.collidepoint(pos):
                        selected_shape = RECTANGLE
                        drawing = False
                        adding_joint = False
                        dragging_object = None
                        dragged_complex = []
                    elif circle_button_rect.collidepoint(pos):
                        selected_shape = CIRCLE
                        drawing = False
                        adding_joint = False
                        dragging_object = None
                        dragged_complex = []
                    elif weld_joint_button_rect.collidepoint(pos):
                        adding_joint = not adding_joint if joint_type == WELD_JOINT else True
                        joint_type = WELD_JOINT
                        selected_shape = None
                        drawing = False
                        dragging_object = None
                        dragged_complex = []
                    elif pivot_joint_button_rect.collidepoint(pos):
                        adding_joint = not adding_joint if joint_type == PIVOT_JOINT else True
                        joint_type = PIVOT_JOINT
                        selected_shape = None
                        drawing = False
                        dragging_object = None
                        dragged_complex = []
                    # Joint creation logic
                    elif adding_joint:
                        # Use world_cursor_pos directly
                        p_world = world_cursor_pos
                        print(f"Joint placement at screen coords {screen_cursor_pos}, world coords {p_world} with camera offset {camera_offset}")
                        search_radius = 15
                        objects_near_click = [] # Store tuples of (distance_sq, object)

                        # Find overlapping objects and calculate distance to click point
                        for obj in shapes: # Iterate through all shapes
                            try:
                                # Use body position as the reference point for distance calculation
                                obj_pos = obj.body.position
                                dist_sq = obj_pos.get_dist_sqrd(p_world)
                                is_close_enough = False

                                # Check if click is within search radius of the object's bounds
                                if isinstance(obj.shape, pymunk.Circle):
                                    # Check distance from center + radius against search radius
                                    if dist_sq <= (obj.shape.radius + search_radius)**2:
                                        is_close_enough = True
                                elif isinstance(obj.shape, pymunk.Poly):
                                    # For polys, check if the click is near the bounding box first for efficiency
                                    # A simple check: distance from center vs rough diagonal + search radius
                                    # This is an approximation but avoids complex point-in-poly checks here
                                    half_width = (obj.shape.bb.right - obj.shape.bb.left) / 2
                                    half_height = (obj.shape.bb.top - obj.shape.bb.bottom) / 2
                                    rough_radius_sq = half_width**2 + half_height**2
                                    if dist_sq <= rough_radius_sq + search_radius**2 * 2: # Heuristic check
                                        # More precise check (optional but better):
                                        info = obj.shape.point_query(p_world)
                                        if info.distance <= search_radius:
                                            is_close_enough = True

                                if is_close_enough:
                                    objects_near_click.append((dist_sq, obj))
                                    print(f"Found object {type(obj).__name__} at {obj_pos}, dist_sq: {dist_sq}")

                            except Exception as e:
                                print(f"Overlap/distance check error: {e}")

                        # Sort the found objects by distance (closest first)
                        objects_near_click.sort(key=lambda item: item[0])

                        print(f"Found {len(objects_near_click)} objects near click, sorted by distance.")

                        if len(objects_near_click) >= 2:
                            # Select the two closest objects
                            body1 = objects_near_click[0][1].body
                            body2 = objects_near_click[1][1].body
                            body1_id = id(body1)
                            body2_id = id(body2)

                            print(f"Selected closest bodies: {type(objects_near_click[0][1]).__name__} and {type(objects_near_click[1][1]).__name__}")

                            # Check if these bodies are already connected
                            already_connected = False
                            for component in connected_components:
                                if body1_id in component and body2_id in component:
                                    already_connected = True
                                    print("Selected bodies are already connected.")
                                    break

                            if not already_connected:
                                # Add the joint using the world click position as the anchor
                                add_joint(space, joints, joint_type, body1, body2, p_world, connected_components)
                                print(f"{joint_type.capitalize()} joint created between bodies near {p_world}")
                                adding_joint = False # Turn off after successful joint creation
                            # else: # Already printed message above
                            #    pass
                        else:
                            print("Need at least two objects near the click point to create a joint.")
                    # Shape drawing start
                    elif selected_shape and not drawing:
                        drawing = True
                        start_pos = screen_cursor_pos  # Store screen position
                        adding_joint = False
                        dragging_object = None
                        dragged_complex = []
                    # Edit mode drag start
                    else:
                        ui_buttons = [start_button_rect, rect_button_rect, circle_button_rect,
                                    weld_joint_button_rect, pivot_joint_button_rect]
                        clicked_ui = any(btn.collidepoint(screen_cursor_pos) for btn in ui_buttons)
                        if not clicked_ui and not adding_joint and not drawing:
                            try:
                                p = world_cursor_pos  # Use tracked world position
                                clicked_object = None
                                # Find the topmost object clicked
                                for obj in reversed(shapes):
                                    try:
                                        # Wake up the body first to prevent sleeping-related errors
                                        if obj.body.is_sleeping:
                                            print(f"Waking up body for drag operation")
                                            obj.body.activate()

                                        # Simple distance-based check for circles
                                        if isinstance(obj.shape, pymunk.Circle):
                                            dist = math.sqrt((obj.body.position.x - p[0])**2 +
                                                            (obj.body.position.y - p[1])**2)
                                            if dist <= obj.shape.radius:
                                                clicked_object = obj
                                                break
                                        # Polygons
                                        elif isinstance(obj.shape, pymunk.Poly):
                                            vertices = [obj.body.local_to_world(v) for v in obj.shape.get_vertices()]
                                            if point_in_polygon(p, vertices):
                                                clicked_object = obj
                                                break
                                    except Exception as e:
                                        print(f"Drag selection error: {e}")

                                if clicked_object:
                                    # Build the complex of connected objects using connected_components
                                    dragged_complex = []  # Reset the complex
                                    clicked_body_id = id(clicked_object.body)

                                    # Find which component contains the clicked object
                                    for component in connected_components:
                                        if clicked_body_id in component:
                                            # Add all objects that belong to this component
                                            for shape in shapes:
                                                if id(shape.body) in component:
                                                    dragged_complex.append(shape)
                                            break

                                    # If not part of any component, just drag the clicked object
                                    if not dragged_complex:
                                        dragged_complex = [clicked_object]

                                    # Store reference to the primary dragged object and offset
                                    offset = clicked_object.body.position - p
                                    dragging_object = clicked_object
                                    drag_offset = offset
                                    print(f"Dragging complex with {len(dragged_complex)} objects")
                                    selected_shape = None
                                    drawing = False
                                    adding_joint = False
                            except Exception as e:
                                print(f"Error starting drag: {e}")
                                # Don't crash on drag errors

                elif game_mode == SIMULATION_MODE:
                    if start_button_rect.collidepoint(screen_cursor_pos):
                        game_mode = EDIT_MODE
                        dragging_object = None # Stop dragging if any
                    else: # Try dragging in sim mode
                        p = world_cursor_pos  # Use tracked world position
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
                            dragging_object = clicked_object_sim
                            drag_offset = offset

            elif event.type == pygame.MOUSEMOTION:
                # Update end_pos for drawing preview
                if drawing:
                    end_pos = screen_cursor_pos

                # Handle dragging with tracked cursor positions
                if dragging_object:
                    p = world_cursor_pos  # Use tracked world position
                    if game_mode == EDIT_MODE:
                        # Use existing dragged_complex or rebuild it if needed
                        if not dragged_complex:
                            dragged_body_id = id(dragging_object.body)
                            for component in connected_components:
                                if dragged_body_id in component:
                                    # Rebuild dragged_complex
                                    dragged_complex = [obj for obj in shapes if id(obj.body) in component]
                                    break

                            # If still empty, just use the main dragged object
                            if not dragged_complex:
                                dragged_complex = [dragging_object]

                        # Calculate the position delta for the primary dragged object
                        target_pos = p + drag_offset
                        delta_pos = target_pos - dragging_object.body.position

                        # Move all objects in the complex by the same delta
                        for obj in dragged_complex:
                            obj.body.position += delta_pos
                            # Make sure to activate the body so it registers position changes
                            obj.body.activate()

                    elif game_mode == SIMULATION_MODE:
                        # Apply velocity to follow mouse
                        target_pos = p + drag_offset
                        # Simple velocity adjustment
                        vel_adj = (target_pos - dragging_object.body.position) * 10
                        dragging_object.body.velocity = vel_adj
                        dragging_object.body.activate()

            elif event.type == pygame.MOUSEBUTTONUP:
                # Use tracked screen cursor position
                pos = screen_cursor_pos

                if dragging_object:
                    # Stop dragging
                    if game_mode == EDIT_MODE:
                        # Put all objects in the complex to sleep
                        for obj in dragged_complex:
                            # Check if body still exists and is part of the space before trying to sleep
                            if obj.body and obj.body.space is space:
                                obj.body.sleep()

                    dragging_object = None
                    drag_offset = None
                    dragged_complex = []  # Clear the complex when done dragging

                elif selected_shape and drawing:
                    # ... (existing shape creation logic - keep this) ...
                    end_pos_up = screen_cursor_pos
                    x1, y1 = from_pygame(start_pos, camera_offset)
                    x2, y2 = from_pygame(end_pos_up, camera_offset)

                    if selected_shape == RECTANGLE:
                        width = abs(x2 - x1)
                        height = abs(y2 - y1)
                        if width > 5 and height > 5:
                            x = min(x1, x2)
                            y = max(y1, y2)  # Pymunk Y is inverted
                            body, shape = create_rectangle(x, y - height, width, height)
                            # Use the add_shape function and store the returned object
                            newly_added_object = add_shape(space, shapes, RECTANGLE, body, shape)
                            # Automatically select the new object
                            selected_shape_object = newly_added_object
                    elif selected_shape == CIRCLE:
                        center_x, center_y = from_pygame(start_pos, camera_offset)
                        radius = math.dist(start_pos, end_pos_up)
                        if radius > 5:
                            body, shape = create_circle(center_x, center_y, radius)
                            # Use the add_shape function and store the returned object
                            newly_added_object = add_shape(space, shapes, CIRCLE, body, shape)
                            # Automatically select the new object
                            selected_shape_object = newly_added_object

                    selected_shape = None # Stop shape selection after creation
                    drawing = False
                    start_pos = (0, 0)
                    end_pos = None # Reset end_pos

            # --- Keyboard Input for Camera ---
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    camera_offset = (camera_offset[0] - camera_speed, camera_offset[1])
                elif event.key == pygame.K_RIGHT:
                    camera_offset = (camera_offset[0] + camera_speed, camera_offset[1])
                elif event.key == pygame.K_UP:
                    camera_offset = (camera_offset[0], camera_offset[1] + camera_speed)
                elif event.key == pygame.K_DOWN:
                    camera_offset = (camera_offset[0], camera_offset[1] - camera_speed)
                elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                    if game_mode == EDIT_MODE:
                        if selected_shape_object:
                            if delete_shape(space, shapes, selected_shape_object):
                                selected_shape_object = None
                                property_buttons = {} # Clear buttons
                        elif selected_joint_object:
                            if delete_joint(space, joints, selected_joint_object, connected_components):
                                selected_joint_object = None
                                joint_menu_buttons = {} # Clear buttons

        if not running: # Check if QUIT event occurred
            break

        # --- State Change Logic ---
        if previous_mode == EDIT_MODE and game_mode == SIMULATION_MODE:
            # Store initial states and apply properties
            print("Switching to SIMULATION mode. Storing initial states.")
            initial_body_states = {} # Clear any previous states
            for obj in shapes:
                try:
                    # Apply mass, elasticity, friction etc. before simulation starts
                    obj.apply_mass() # Calculate mass/moment based on shape/density
                    obj.shape.elasticity = obj.elasticity
                    obj.shape.friction = obj.friction

                    # Store state: Position, Angle, and ZERO velocities
                    body_state = {
                        'position': pymunk.Vec2d(obj.body.position.x, obj.body.position.y),
                        'angle': obj.body.angle,
                        'velocity': pymunk.Vec2d(0, 0), # Explicitly store zero velocity
                        'angular_velocity': 0.0         # Explicitly store zero angular velocity
                    }
                    
                    # Store in both dictionaries
                    initial_body_states[id(obj.body)] = body_state
                    original_states[id(obj.body)] = body_state
                    
                    # Ensure bodies are awake for simulation start
                    obj.body.activate()
                except Exception as e:
                    print(f"Error storing state or activating body {id(obj.body)}: {e}")

            # Store joint states if needed in the future
            for joint in joints:
                try:
                    if joint.body_a: joint.body_a.activate()
                    if joint.body_b: joint.body_b.activate()
                except Exception as e:
                    print(f"Error activating joint bodies: {e}")


        elif previous_mode == SIMULATION_MODE and game_mode == EDIT_MODE:
            # Restore initial states
            print(f"Switching to EDIT mode. Restoring {len(original_states)} original states.")
            restore_count = 0
            
            for obj in shapes:
                body_id = id(obj.body)
                if body_id in original_states:
                    state = original_states[body_id]
                    # Use safe operations to restore position, angle, and ZERO velocities
                    try:
                        # Ensure the body is awake for position change
                        obj.body.activate()
                        
                        # Restore exact position and orientation
                        obj.body.position = pymunk.Vec2d(state['position'].x, state['position'].y)
                        obj.body.angle = state['angle']
                        
                        # Explicitly zero out all velocities (linear and angular)
                        obj.body.velocity = pymunk.Vec2d(0, 0)
                        obj.body.angular_velocity = 0
                        
                        # Put body to sleep after resetting
                        obj.body.sleep()
                        restore_count += 1
                    except Exception as e:
                        print(f"Error restoring body {body_id}: {e}")
                else:
                    # If a shape was somehow added during simulation
                    print(f"Warning: No original state found for body {body_id}. Attempting to sleep.")
                    try:
                        obj.body.velocity = pymunk.Vec2d(0, 0)
                        obj.body.angular_velocity = 0
                        obj.body.sleep()
                    except Exception as e:
                        print(f"Error handling new body: {e}")
            
            print(f"Successfully restored {restore_count}/{len(shapes)} objects to original positions")
            
            # Keep the original_states dictionary for future restorations
            # But clear the temporary one used during this simulation run
            initial_body_states = {}
            
            # Ensure selection is cleared and dragging stops
            selected_shape_object = None
            selected_joint_object = None
            property_buttons = {}
            joint_menu_buttons = {}
            dragging_object = None
            dragged_complex = []

        # --- Drawing ---
        screen.fill(SKY_BLUE) # Background

        # Draw continuous ground based on camera
        draw_continuous_ground(screen, camera_offset)

        # Draw all shapes with camera offset
        for obj in shapes:
            # Apply visual selection highlight in EDIT mode
            if game_mode == EDIT_MODE and obj == selected_shape_object:
                # Create a temporary surface for the shape
                temp_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                obj.draw(temp_surf, camera_offset) # Draw shape onto temp surface

                # Create a highlight color (e.g., yellow with alpha)
                highlight_color = (255, 255, 0, 100) # Yellow, semi-transparent

                # Fill the shape area on the temp surface with the highlight color using BLEND_RGBA_MULT
                # This requires finding the pixels belonging to the shape, which is complex.
                # A simpler approach: Draw a slightly larger outline or overlay.
                # Let's draw a yellow outline for simplicity
                if isinstance(obj, Rectangle):
                    local_verts = obj.shape.get_vertices()
                    world_verts = [obj.body.local_to_world(v) for v in local_verts]
                    pygame_verts = [to_pygame(v, camera_offset) for v in world_verts]
                    pygame.draw.polygon(screen, (255, 255, 0), pygame_verts, 3) # Yellow outline
                elif isinstance(obj, Circle):
                    position = to_pygame(obj.body.position, camera_offset)
                    pygame.draw.circle(screen, (255, 255, 0), position, int(obj.radius) + 2, 3) # Yellow outline

            # Draw the actual shape (on top of highlight if selected)
            obj.draw(screen, camera_offset)

        # Draw all joints with camera offset
        for joint in joints:
            # Highlight selected joint
            if game_mode == EDIT_MODE and joint == selected_joint_object:
                 # Draw a thicker/different color representation for the selected joint
                 # Example: Draw a larger circle or square at the joint point
                 if isinstance(joint, PivotJoint) or isinstance(joint, WeldJoint):
                     if joint.body_a:
                         current_joint_world_pos = joint.body_a.local_to_world(joint.anchor_a_local)
                         joint_pos_pygame = to_pygame(current_joint_world_pos, camera_offset)
                         pygame.draw.circle(screen, (255, 0, 255), joint_pos_pygame, 8, 3) # Magenta highlight circle
            # Draw the regular joint representation
            joint.draw(screen, camera_offset)

        # Draw shape being created (uses screen coordinates, no offset needed)
        if drawing and end_pos:
            draw_shape_being_created(screen, selected_shape, drawing, start_pos, end_pos)

        # Draw UI (fixed position, no camera offset)
        draw_ui(screen, game_mode, start_button_rect, start_button_text, stop_button_text,
                weld_joint_button_rect, weld_joint_button_text,
                pivot_joint_button_rect, pivot_joint_button_text,
                rect_button_rect, rect_button_text,
                circle_button_rect, circle_button_text,
                adding_joint, joint_type, camera_offset) # Pass camera offset for display

        # Draw properties menu if an object is selected in EDIT mode
        if game_mode == EDIT_MODE:
            if selected_shape_object:
                property_buttons = draw_properties_menu(screen, selected_shape_object, font)
            elif selected_joint_object:
                joint_menu_buttons = draw_joint_menu(screen, selected_joint_object, font)

        # Draw custom cursor if adding pivot joint
        if show_custom_cursor:
            pygame.draw.circle(screen, BLACK, screen_cursor_pos, 8, 1) # Draw a circle cursor

        # --- Simulation Step ---
        if game_mode == SIMULATION_MODE:
            dt = 1.0 / FPS
            space.step(dt)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

# --- Helper Functions (Keep as is) ---
def point_in_polygon(point, polygon_vertices):
    """Checks if a point is inside a polygon using the ray casting algorithm."""
    x, y = point
    n = len(polygon_vertices)
    inside = False
    p1x, p1y = polygon_vertices[0]
    for i in range(n + 1):
        p2x, p2y = polygon_vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

if __name__ == '__main__':
    run_simulation()