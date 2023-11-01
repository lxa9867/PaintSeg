import cv2
import numpy as np

def sample_edge_points(mask, num_points=15):
    index = np.where(mask == 1)
    index = np.array(index)
    bottom, top = np.min(index[0]), np.max(index[0])
    values = np.linspace(bottom, top, num_points)
    sampled_points =[]
    count = 0
    for i, v in enumerate(values):
        v = int(v)
        index_x = index[1, index[0, :] == v]
        if len(index_x) == 0:
            continue
        if count % 2 == 0:
            sampled_points.append((np.min(index_x), v))
        else:
            sampled_points.append((np.max(index_x), v))
        count += 1

    return sampled_points

def generate_scribble(mask, num_points=30):
    # Get the evenly distributed points in the mask
    point_coords = sample_edge_points(mask, num_points)
    # Create a blank image to draw the scribble on
    scribble = np.zeros(mask.shape, dtype=np.uint8)

    # Connect the randomly selected points with lines
    for i in range(len(point_coords) - 1):
        start = point_coords[i]
        end = point_coords[i+1]
        cv2.line(scribble, start, end, color=1, thickness=15)
    scribble = scribble * mask
    # Return the scribble image
    return scribble

if __name__ == '__main__':
    mask = cv2.imread(f'/Users/xiangli/Downloads/00005.png')
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    # scribble = generate_scribbles(mask, 10, 5)
    # _, mask = get_evenly_distributed_points(mask, 10)
    mask = generate_scribble(mask, num_points=30)
    cv2.imshow('123', mask)
    cv2.waitKey(0)