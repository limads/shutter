/* The edge segmenter can start at equally-spaced points and extend rays from them until
they fuse or an edge is met. The patch is defined by the points where the rays meet the
edges, generated for example with a canny algorithm. */

/*
pub struct HoughSegmenter {

}

pub struct Hough {

}

After Ballard & Brown (p. 123), to detect lines:

(1) Determine a range of admissible values for line equation parameters c and m (c = mx + y)
(2) Form an accumulator A(c, m) set to zero
(3) For each point in a gradient image above a threshold, increment all points in the accumulator array along th eappropriate line
(4) Local maxima of the accumulator are collinear points in the image array. The values at the accumulator gives how many points there are in a line.

This method can be generalized for curves of higher order.

To avoid infinite slopes, lines can be parametrized by r (distance from origin) and theta (angle between normal to line to x-axis).
r = x cos theta + y sin theta

*/


