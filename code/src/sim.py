from gsim.src.model import *
from gsim.src.controllers import *
import numpy as np
import pygame
import random
import yaml
import sys
import os 

def render_textrect(string, font, rect):
    final_lines = []
    requested_lines = string.splitlines()

    for requested_line in requested_lines:
        if font.size(requested_line)[0] > rect.width:
            words = requested_line.split(' ')
            
            final_words = []
            for word in words:
                if font.size(word)[0] >= rect.width:
                    accumulated_word = ''
                    for c in word:
                        test_word = accumulated_word + c
                        if font.size(test_word)[0]+font.size('-')[0] < rect.width:
                            accumulated_word = test_word
                        else:
                            final_words.append(accumulated_word+'-')
                            accumulated_word = c
                    final_words.append(accumulated_word)
                else:
                    final_words.append(word)
            
            accumulated_line = ''
            for word in final_words:
                test_line = accumulated_line + word + ' '
                if font.size(test_line)[0] < rect.width:
                    accumulated_line = test_line
                else:
                    final_lines.append(accumulated_line)
                    accumulated_line = word + ' '
            final_lines.append(accumulated_line)
        else:
            final_lines.append(requested_line)

    surface = pygame.Surface(rect.size)
    surface.fill((255,255,255))

    accumulated_height = 0
    for line in final_lines:
        if accumulated_height + font.size(line)[1] >= rect.height:
            return surface, False
        
        if line != '':
            tempsurface = font.render(line, True, (0,0,0))
            surface.blit(tempsurface, (0, accumulated_height))

        accumulated_height += font.size(line)[1]

    return surface, True

class GSim(object):
    def __init__(self, width, height, frame_files):
        pygame.init()
        self.canvas = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        self.slide = 0
        self.numFramesDone = 0
        self.enter = False
        self.text = ''
        self.maxTextLen = 50

        self.instructionsText = "  You will be teaching a robot to perform a simple task by demonstrating it in simulation and providing a written instruction. Specifically:\n\n - The task will consist of moving a shape to a specified location.\n\n - The shape to be moved will be on the right, and the location to move it to will be marked by an 'x'.\n\n - Upon completion of the demonstrations, you will be asked to provide a simple one-sentence instruction that you believe would teach a robot to complete the task. Assume the robot has the intelligence of a child.\n\n - The robot cannot see the 'x' in the scenes, so do not reference it in your sentence. The robot can see all of the shapes.\n\n - There is no correct sentence to write down. Just write a sentence that you think would teach a robot how to complete the task you just demonstrated.\n\n\n               Press 'Enter' to continue"
        self.taskInstructionText = "  Please provide a simple one-sentence instruction that you believe would teach a robot to complete the task. Assume the robot has the intelligence of a child. Remeber that the robot cannot see the 'x'.\n\nYou can use the left and right arrow keys to view the scenes. Close the window when you are done."

        self.model = GSimModel(width, height)
        self.frame_files = frame_files
        for f in frame_files:
            self.model.addFrameFromFile(f)

        self.font = pygame.font.SysFont("monospace", 30)
        self.enterText = self.font.render("Press 'Enter' to continue", True, (0,0,0))

        self.instructionsRect = pygame.Rect((width/10, height/10, 8*width/10, 8*height/10))
        self.taskInstructionRect = pygame.Rect((width/10, height/10, 8*width/10, 3*height/10))
        self.textRect = pygame.Rect((width/10, 4*height/10, 8*width/10, height/10))

    def clicked(self, obj, x, y):
        if isinstance(obj, Shape) and not isinstance(obj, Target) and not isinstance(obj, Stage) and not isinstance(obj, Table):   
            return x > obj.x and x < obj.x + obj.w and y > obj.y and y < obj.y + obj.w
        elif isinstance(obj, Gripper):
            return x > obj.x - obj.r and x < obj.x + obj.r and y > obj.y - obj.r and y < obj.y + obj.r
        else:
            return False

    def done(self):
        return self.numFramesDone == self.model.getNumFrames()

    def next(self):
        if self.slide == 0:
            self.slide = 1

        elif self.slide == 1:
            if not self.model.getCurrentFrame().isActive():
                if self.model.getCurrentFrameId() == self.model.getNumFrames()-1:
                    self.slide = 2

                self.model.upFrame()

                if self.numFramesDone < self.model.getNumFrames():
                    self.numFramesDone += 1

    def back(self):
        if self.slide == 1:
            self.model.downFrame()

        elif self.slide == 2:
            self.slide = 1

    def addText(self, event):
        if self.slide == 2:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                self.text += '\n'
            else:
                 self.text += event.unicode

    def getUserPath(self, path, uid):
        return path + '/user_' + str(uid)

    def getExperimentPath(self, path, uid, eid):
        return self.getUserPath(path, uid) + '/experiment_' + str(eid)

    def mkdirs(self, path, uid, eid):
        e_path = self.getExperimentPath(path, uid, eid)
        if not os.path.exists(e_path):
            os.makedirs(e_path)

    def saveWorlds(self, path, uid, eid):
        filename = self.getExperimentPath(path, uid, eid) + '/worlds.yaml'
        yaml.dump({'worlds': self.frame_files}, open(filename, 'w'))

    def saveMotion(self, path, uid, eid):
        filename = self.getExperimentPath(path, uid, eid) + '/motion.npy'
        np.save(filename, self.model.getMotion())

    def saveText(self, path, uid, eid):
        filename = self.getExperimentPath(path, uid, eid) + '/task.txt'
        f = open(filename, 'w')
        f.write(self.text)

    def update(self):
        self.canvas.fill((255,255,255))

        if self.slide == 0:
            rendered_text, fits = render_textrect(self.instructionsText, self.font, self.instructionsRect)
            self.canvas.blit(rendered_text, self.instructionsRect.topleft)
        elif self.slide == 1:
            self.model.draw(self.canvas)

            if self.enter:
                self.canvas.blit(self.enterText, (10, 10))
        else:
            rendered_text, fits = render_textrect(self.taskInstructionText, self.font, self.taskInstructionRect)
            self.canvas.blit(rendered_text, self.taskInstructionRect.topleft)

            x, y, w, h = self.textRect.x, self.textRect.y, self.textRect.w, self.textRect.h
            rendered_text, fits = render_textrect(self.text, self.font, self.textRect)
            self.canvas.blit(rendered_text, self.textRect.topleft)
            pygame.draw.rect(self.canvas, (0,0,0), [x-10, y-10, w+20, h+20], 1)
            
        pygame.display.flip()
        
    def run(self, path, uid, eid):
        mouse_pos = (0, 0)
        t = 0

        while True:
            dt = self.clock.tick(50)
            t += dt
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.mkdirs(path, uid, eid)
                    self.saveWorlds(path, uid, eid)
                    self.saveMotion(path, uid, eid)
                    self.saveText(path, uid, eid)
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    x, y = mouse_pos
                    
                    if self.done():
                        if event.key == pygame.K_RIGHT:
                            self.next()

                        elif event.key == pygame.K_LEFT:
                            self.back()

                        else:
                            self.addText(event)
                    else:
                        if event.key == pygame.K_RETURN:
                            self.next()
                            self.enter = False

                if self.slide == 1:
                    if event.type == pygame.MOUSEMOTION:
                        x, y = event.pos

                        self.model.moveGripper(x, y)
                        if not self.done() and not self.model.getCurrentFrame().isActive():
                            self.enter = True

                        mouse_pos = (x, y)

                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
            
                        if event.button == 1:
                            self.model.closeGripper()

                self.update()

class GSimPlayback(object):
    def __init__(self, width, height, frame, policy):
        pygame.init()
        self.canvas = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        self.model = GSimModel(width, height)
        self.model.addFrameFromFile(frame)

        self.policy = policy

    def update(self):
        self.canvas.fill((255,255,255))
        self.model.draw(self.canvas)
        pygame.display.flip()

    def run(self, start, goal):
        X = self.policy.playback(start, goal)
        i = 0

        while True:
            self.clock.tick(50)

            x, y = X[i,:]
            if i == 0:
                self.model.closeGripper()
            else:
                self.model.moveGripper(x, y)
            
            if i < len(X)-1:
                i += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            self.update()
 
if __name__ == '__main__':
    if len(sys.argv) >= 5:
        worlds_path = sys.argv[1]
        save_path = sys.argv[2]
        uid = int(sys.argv[3])
        eid = int(sys.argv[4])

        worlds = yaml.load(open(worlds_path))['worlds']
        worlds = ['../worlds/'+w for w in worlds]

        sim = GSim(1300, 1000, worlds)
        sim.run(save_path, uid, eid)

    '''
    sim = GSim(1300, 1000, ['../worlds/w1.yaml', '../worlds/w2.yaml', '../worlds/w3.yaml'])

    if len(sys.argv) == 1:
        sim.run('../data', 0, 1)
    else:
        sim.run(sys.argv[1])
    '''
