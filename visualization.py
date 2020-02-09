import pygame
import pygame.gfxdraw
from pygame.locals import *
import sys
import random
# from project_St import audio_analysis
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


pygame.init()
CLOCK = pygame.time.Clock()


''' DISPLAY SETUP -------------------------------------------------------------------------------- DISPLAY SETUP '''
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
DW_HALF = DISPLAY_WIDTH / 2
DH_HALF = DISPLAY_HEIGHT / 2
DISPLAY_AREA = DISPLAY_WIDTH * DISPLAY_HEIGHT
DS = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
rect_ds = DS.get_rect()
print(rect_ds)

''' LOAD IMAGES ---------------------------------------------------------------------------------- LOAD IMAGES '''
background_source = pygame.image.load('images/spaceship.jpg')
rect_bg_source = background_source.get_rect()
scale_bg = min(rect_bg_source.width / rect_ds.width, rect_bg_source.height / rect_ds.height)
print(rect_bg_source.height / scale_bg)
# background = pygame.transform.scale(background_source, (int(rect_bg_source.width / scale_bg),
#                                                         int(rect_bg_source.height / scale_bg)))
background = background_source
rect_bg = background.get_rect()

particle_l1 = pygame.image.load('images/stardust1.png')
particle_l2 = pygame.image.load('images/stardust2.png')
rect_pc_l = particle_l1.get_rect()

particle_s1 = pygame.transform.scale(particle_l1, (200, 100))
particle_s2 = pygame.transform.scale(particle_l2, (200, 100))
rect_pc_s = particle_s1.get_rect()

locator = pygame.image.load('images/locator.png')
rect_locator = locator.get_rect()
# FUNCTIONS ------------------------------------------------------------------------------------------------ FUNCTIONS
def event_handler():
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit()
            sys.exit()


''' SETUP VARIABLES ------------------------------------------------------------------------------ SETUP VARIABLES '''
MAX_SCALE = 1380
MIN_SCALE = 1290
current_scale = MIN_SCALE
scale_direction = 2
pc_pos = [0, 100, 200]
pc_scale = [4, 6, 8]

bd = 1
bc = 0

'''create frames ---------------------------------------------------------------------------'''

engine_pos_l = (480, 340)
engine_pos_r = (620, 225)
engine_pos_3 = (1080, 460)

'''load music -----------------------------------------------------------------------------'''


if __name__ == '__main__':

    script_df = pd.read_csv('songs/Accelerator.csv')
    script_onset = script_df.to_numpy()[:, 1]
    script_onset = np.roll(script_df.to_numpy()[:, 1], -1)
    script_energy = script_df.to_numpy()[:, 2]
    script_pc = script_df.to_numpy()[:, 0]
    script_pc = gaussian_filter(script_energy, sigma=5)
    bpm = script_df['global_bpm'][0]
    script_bright = np.roll(gaussian_filter(script_energy, sigma=5) * 3, -2)
    init_frame = True
    run = True
    frame_counter = 0
    scaling = True
    vibrating = True
    brightening = True
    darkening = True
    particles = True
    fps_delay = 100
    frame_index = 0
    scaling_step = 6
    scaling_frame = []
    for scale in range(1400, 1400 + int(bpm * 16 * scaling_step / 60) + 1):
        scaled_bg = pygame.transform.scale(background, (scale, int(scale / 16 * 9)))
        scaling_frame.append(scaled_bg)
    frame_len = len(scaling_frame)

    vibration_l = -2
    vibration_h = 2
    vibration_x = 0
    vibration_y = 0
    vibration_counter = 0
    brighten = 0
    brighten_step = 64
    darken = 32
    darken_step = 1
    pc_init_frame = True
    pc_vib_l = 0
    pc_vib_h = 10
    pc_pos_1 = [None, None, None]
    pc_pos_2 = [None, None, None]
    pc_pos_3 = [None, None, None]
    pc_step_base = np.array([10, 10, 10])

    pygame.mixer.music.load("songs/Accelerator.ogg")
    pygame.mixer.music.play()
    while run:
        CLOCK.tick()
        event_handler()
        bg = scaling_frame[frame_index]
        rect_bg_n = bg.get_rect()

        if vibrating:
            if script_onset[frame_counter] == 1:
                vibration_counter = 1
            if 3 > vibration_counter > 0:
                vibration_x = random.randint(vibration_l, vibration_h) * (1 + int(script_pc[frame_counter] * 5))
                vibration_y = random.randint(vibration_l, vibration_h) * (1 + int(script_pc[frame_counter] * 5))
                vibration_counter += 1
            else:
                vibration_x = 0
                vibration_y = 0
                vibration_counter = 0

        if scaling:
            if frame_index + scaling_step >= frame_len or frame_index + scaling_step < 0:
                scaling_step = - scaling_step
            frame_index = frame_index + scaling_step

        if brightening:
            brighten = brighten_step * script_bright[frame_counter]

        if darkening:
            darken = darken_step

        DS.blit(bg, (DW_HALF - rect_bg_n.center[0] + vibration_x,
                     DH_HALF - rect_bg_n.center[1] + vibration_y))

        if particles and script_pc[frame_counter] > 0.05:
            pc_step = pc_step_base * (script_pc[frame_counter]) * 20
            for pc_pos_index in range(len(pc_pos_1)):
                if pc_pos_1[pc_pos_index] is None:
                    pc_pos_1[pc_pos_index] = (engine_pos_l[0] - rect_pc_l.width + random.randint(pc_vib_l, pc_vib_h),
                                              engine_pos_l[1] - rect_pc_l.height + random.randint(pc_vib_l, pc_vib_h))

            for pc_pos_index in range(len(pc_pos_2)):
                if pc_pos_2[pc_pos_index] is None:
                    pc_pos_2[pc_pos_index] = (engine_pos_r[0] - rect_pc_l.width + random.randint(pc_vib_l, pc_vib_h),
                                              engine_pos_r[1] - rect_pc_l.height + random.randint(pc_vib_l, pc_vib_h))

            for pc_pos_index in range(len(pc_pos_3)):
                if pc_pos_3[pc_pos_index] is None:
                    pc_pos_3[pc_pos_index] = (engine_pos_3[0] - rect_pc_l.width + random.randint(pc_vib_l, pc_vib_h),
                                              engine_pos_3[1] - rect_pc_l.height + random.randint(pc_vib_l, pc_vib_h))
            if pc_init_frame:
                pc_pos_1[1] = (pc_pos_1[1][0],
                               pc_pos_1[1][1] + int((rect_ds.height - pc_pos_1[1][1]) / 2))
                pc_pos_2[1] = (pc_pos_2[1][0] + int((rect_ds.width - pc_pos_2[1][0]) / 2),
                               pc_pos_2[1][1])
                pc_pos_3[1] = (pc_pos_3[1][0] + int((rect_ds.width - pc_pos_3[1][0]) / 2),
                               pc_pos_3[1][1])
                pc_init_frame = False
            DS.blit(particle_l1, pc_pos_1[0])
            DS.blit(particle_l2, pc_pos_1[1])
            DS.blit(particle_l1, pc_pos_2[0])
            DS.blit(particle_l2, pc_pos_2[1])
            DS.blit(particle_s1, pc_pos_3[0])
            DS.blit(particle_s2, pc_pos_3[1])
            for pc_pos_index in range(len(pc_pos_1)):
                pc_pos_1[pc_pos_index] = (pc_pos_1[pc_pos_index][0] - int(pc_step[pc_pos_index] / 6),
                                          pc_pos_1[pc_pos_index][1] + pc_step[pc_pos_index])
                if pc_pos_1[pc_pos_index][1] > rect_ds.height:
                    pc_pos_1[pc_pos_index] = None
            for pc_pos_index in range(len(pc_pos_2)):
                pc_pos_2[pc_pos_index] = (pc_pos_2[pc_pos_index][0] + pc_step[pc_pos_index],
                                          pc_pos_2[pc_pos_index][1] - int(pc_step[pc_pos_index] / 6))
                if pc_pos_2[pc_pos_index][0] > rect_ds.width:
                    pc_pos_2[pc_pos_index] = None
            for pc_pos_index in range(len(pc_pos_3)):
                pc_pos_3[pc_pos_index] = (pc_pos_3[pc_pos_index][0] + pc_step[pc_pos_index],
                                          pc_pos_3[pc_pos_index][1] + int(pc_step[pc_pos_index] / 3))
                if pc_pos_3[pc_pos_index][0] > rect_ds.width:
                    pc_pos_3[pc_pos_index] = None
        else:
            pc_init_frame = True

        DS.fill((darken, darken, darken), special_flags=pygame.BLEND_RGB_SUB)
        DS.fill((brighten, brighten, brighten), special_flags=pygame.BLEND_RGB_ADD)
        init_frame = False
        frame_counter = frame_counter + 1
        pygame.display.update()
        pygame.time.wait(fps_delay - 41)
        CLOCK.tick()
        # print(CLOCK.get_time())
        # clear the display surface to black
