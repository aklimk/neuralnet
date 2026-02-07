"""Utilities to create and prepare network input images for inference.

The network supports 28x28, greyscale, inverted color images.
The MNIST example program supports input arrays in IDX format only.
The program gives a simple, non-erasable drawing surface then saves it
in IDX file fomatting, ready for network inference.
"""

import math
import struct

import numpy
import pygame
from PIL import Image


def ConvertToIDX(image_path="image.png", output_path="image.idx"):
    """Converts a raw image to the formatt neccecary for network inference.

    Downsacles the image to 28x28, inverts the colors and saves the resulting
    array as an IDX format, which the network can read and classify.

    """
    image = Image.open(image_path)
    image = image.resize((28, 28)).convert("L")
    canvas = numpy.array(image.getdata())

    # Invert colors. Needed for compliance with MNIST.
    canvas = 255 - canvas

    # Save image to flat array in IDX format.
    with open(output_path, "wb") as file:
        # 2 bytes : padding
        # 1 byte : data type
        # 1 byte : dimensions
        file.write(struct.pack(">BBBB", 0, 0, 0x08, 1))

        # 4 bytes : dimension 1 size
        file.write(struct.pack(">I", 784))

        for byte in canvas.flatten():
            file.write(struct.pack(">B", byte))


def GetDrawing(size=600, output_path="image.png"):
    """Creates a basic greyscale drawing surface with a single pen style and no eraser.

    Saves the resulting drawing to a file.
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

            # Pen tip is a fixed width black circle.
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

    # Save image to file.
    image = Image.fromarray(numpy.swapaxes(canvas, 0, 1))
    image.save(output_path)


if __name__ == "__main__":
    GetDrawing()
    ConvertToIDX()
