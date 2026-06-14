import pygame
import pymunk

from pygame.color import THECOLORS

GOAL_OBJECT_COLOR = THECOLORS["blue"]
MAP_COLOR = THECOLORS["black"]


def _to_pygame(p, screen_height):
    # [ZMIANA] Własna, bezpieczna funkcja konwersji współrzędnych (Pymunk używa Y rosnącego w górę, Pygame w dół)
    # Zastępuje przestarzałe i problematyczne pymunk.pygame_util.to_pygame z nowej wersji Pymunk
    return pymunk.Vec2d(p[0], screen_height - p[1])


def draw_thresholds(screen, clusters, offset, screen_size):
    for cluster in clusters:
        x = cluster.threshold.position + offset[0]
        points = [(x, 0), (x, screen_size[1])]
        # [ZMIANA] Konwersja koloru z numpy.ndarray na krotkę (tuple), aby uniknąć błędów w nowym Pygame
        pygame.draw.lines(screen, tuple(cluster.color), False, points)


def draw_enemy(screen, position, offset, screen_size):
    points = [(position + offset[0], 0), (position + offset[0], screen_size[0])]
    try:
        wand_img = pygame.image.load('assets/wand.png')
        screen.blit(wand_img, (position + offset[0] - 31, 0))
    except pygame.error:
        # Fallback jeśli obrazka nie ma
        pygame.draw.line(screen, THECOLORS["red"], points[0], points[1], 5)


def draw_clusters(screen, clusters, offset):
    h = screen.get_height()
    for cluster in clusters:
        for bot in cluster.bots:
            # [ZMIANA] Użycie naszej bezpiecznej funkcji _to_pygame
            pos = _to_pygame(bot.body.position, h)
            x = int(pos.x + offset[0])
            y = int(pos.y + offset[1])
            # [ZMIANA] Konwersja koloru numpy na tuple oraz rzutowanie pozycji na int
            pygame.draw.circle(screen, tuple(bot.color), (x, y), int(bot.radius))


def draw_map(screen, map_segment, map_width, map_offset):
    h = screen.get_height()
    for fragment in map_segment:
        p1 = _to_pygame(fragment.a, h)
        p2 = _to_pygame(fragment.b, h)

        mo_x, mo_y = map_offset

        # [ZMIANA] Bezpieczne obliczanie współrzędnych start/end by uniknąć błędów wektorów typu Vec2d
        start_pos = (p1.x - mo_x, p1.y - mo_y)
        end_pos = (p2.x - mo_x, p2.y - mo_y)

        pygame.draw.line(screen, MAP_COLOR, start_pos, end_pos, 2 * map_width)


def draw_goal_object(screen, goal_object, screen_size):
    body = goal_object.body

    # [ZMIANA] Ręczne obliczanie pozycji prostokąta, by zgrać to ze śledzącą kamerą (offset)
    # Stary kod wymuszał rysowanie na środku ekranu, co maskowało fakt spadania obiektu.
    vertices = goal_object.get_vertices()
    ps = []
    for v in vertices:
        v_rot = v.rotated(body.angle)
        px = int(v_rot.x + screen_size[0] // 2)
        py = int(-v_rot.y + screen_size[1] // 2)
        ps.append((px, py))

    pygame.draw.polygon(screen, GOAL_OBJECT_COLOR, ps)