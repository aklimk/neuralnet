import pygame
import numpy
import math
from PIL import Image
import struct

def GetDrawing(size=600):
    """
    Creates a basic greyscale drawing surface with a single pen style and no eraser.

    Pixel values are returned from the function.
    """
    pygame.init()

    # Create
    screen = pygame.display.set_mode((size, size))
    pygame.display.set_caption("Drawing Pad")
    screen.fill((255, 255, 255))
    
    drawing = False
    running = True
    while running:
        if drawing:
            # Draw pixels at mouse position.
            mouse_x, mouse_y = pygame.mouse.get_pos()
            pygame.draw.circle(screen, 0, (mouse_x, mouse_y), size // 25)

        # Detect mouse down and gui events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drawing = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False

        # Update pygame display.
        pygame.display.flip()

    # Save pygame screen to array.
    canvas = pygame.surfarray.array2d(screen)
    # Return array when drawing is finished.
    pygame.quit()
    return canvas




if __name__ == "__main__":
    # Get raw drawing data from drawing pad.
    canvas = GetDrawing()

    # Resize image and convert to greyscale using PIL.
    image = Image.fromarray(numpy.swapaxes(canvas, 0, 1))
    image = image.resize((28, 28)).convert('L')
    canvas = numpy.array(image.getdata())
    
    # Invert colors. Needed for compliance with MNIST.
    canvas = 255 - canvas

    # Save image to flat array in IDX format.
    with open("image.idx", "wb") as file:
        # 2 bytes : padding
        # 1 byte : data type
        # 1 byte : dimensions
        file.write(struct.pack(">BBBB", 0, 0, 0x08, 1))

        # 4 bytes : dimension 1 size
        file.write(struct.pack(">I", 784))

        for byte in canvas.flatten():
            file.write(struct.pack(">B", byte))
    
