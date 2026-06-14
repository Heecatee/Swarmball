import random
import numpy
import pymunk

try:
    from . import simulation_utils as utils
    from . import generate_map as gen
except ImportError:
    from . import simulation_utils as utils
    from . import generate_map as gen

# ZMIANA: Fizyka zoptymalizowana pod pchanie kwadratu przez nierówny teren
ELASTICITY = 0.1            # było 0.3 - mniej "odbijania" = bardziej płynny ruch
FRICTION = 12               # było 9 - więcej tarcia BOT<->TEREN = lepsze "wspinanie"
BOTS_RADIUS = 5
BOTS_MASS = 15              # było 23 - lżejsze boty = szybciej się ruszają i reagują
GOAL_OBJECT_SIZE = (20, 20)
GOAL_OBJECT_MASS = 20       # było 30 - lżejszy kwadrat = łatwiej go pchnąć
GOAL_OBJECT_FRICTION = 0.05 # było 0.01 - trochę więcej tarcia = nie ześlizguje się z pochyłości
SEGMENTS_PER_DIFFICULTY = 3


def create_clusters(number_of_clusters, screen_size, number_of_bots_per_threshold):
    clusters = []
    for cluster_nr in range(number_of_clusters):
        color = numpy.array([100, 100, 100])
        color[cluster_nr % color.shape[0]] += (cluster_nr + 1) * 37
        color[cluster_nr % color.shape[0]] = color[cluster_nr % color.shape[0]] % 256
        threshold = utils.Threshold(
            position=random.randint(-screen_size[0] // 6, screen_size[0] // 6),
            velocity=0
        )
        cluster = utils.Cluster(color, threshold, bots=[])
        for _ in range(number_of_bots_per_threshold):
            cluster.bots.append(create_bot(threshold.position, color, screen_size))
        clusters.append(cluster)
    return clusters


def create_bot(position_x, color, screen_size, max_distance_from_threshold=100):
    min_pos = position_x - max_distance_from_threshold
    max_pos = position_x + max_distance_from_threshold

    radius = BOTS_RADIUS
    mass = BOTS_MASS
    body = pymunk.Body(
        mass,
        moment=pymunk.moment_for_circle(mass, inner_radius=0, outer_radius=radius)
    )
    body.position = (random.randint(min_pos, max_pos), 20)
    shape = pymunk.Circle(body, radius)
    shape.color = color
    shape.elasticity = ELASTICITY
    shape.friction = FRICTION
    shape._body_to_keep_alive = body
    return shape


def create_goal_object(position):
    mass = GOAL_OBJECT_MASS
    size = GOAL_OBJECT_SIZE
    inertia = pymunk.moment_for_box(mass, size)
    body = pymunk.Body(mass, inertia)
    body.position = position
    shape = pymunk.Poly.create_box(body, size)
    shape.elasticity = ELASTICITY
    shape.friction = GOAL_OBJECT_FRICTION
    shape._body_to_keep_alive = body
    return shape


def create_map_segment(difficulty, space, starting_point, segment_size, map_width, segment_count):
    diff = difficulty
    if not diff:
        diff = segment_count // SEGMENTS_PER_DIFFICULTY + 1
        if diff > 6:
            diff = 6
        diff = gen.Difficulty(diff)

    map_fragments = gen.generate_map(
        diff_level=diff,
        x_offset=starting_point[0],
        y_offset=starting_point[1],
        resolution=segment_size
    )
    fragment_start = map_fragments[0]
    map_segment = []
    for fragment_end in map_fragments[1:]:
        fragment = pymunk.Segment(
            space.static_body, tuple(fragment_start), tuple(fragment_end), map_width
        )
        fragment_start = fragment_end
        fragment.elasticity = ELASTICITY
        fragment.friction = FRICTION
        map_segment.append(fragment)
    segment_end_point = fragment_end
    return map_segment, segment_end_point