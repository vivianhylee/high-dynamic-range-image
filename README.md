# high-dynamic-range-image
Creating HDR image from image stack with multiple exposures

## Introduction
The goal of this project is to recover high dynamic range radiance maps from photographs in order to crease an image that captures details from the entire dynaimic range. We first implement algorithm from [Debevec, Malik](http://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf) to recover high dynamic range radiance map and then apply tone mapping and intensity adjustment to convert the radiance map into displayable HDR image.

## Algorithm Overview
### High Dynamic Range Radiance Map Construction
1. Film Response Curve Recovery
>Film response curve is a function g=ln(f-1) maps from pixel values (from 0 to 255) to the log of exposure values: g(Zij) = ln(Ei) + ln(tj) (equation 2 in Debevec). Zij is the observed pixel value at pixel i from image j of image stack and it is a function of scene radiance and known exposure duration, Zij = f(Ei * Δtj). Ei is the unknown scene radiance at pixel i, and scene radiance integrated over some time (Ei * Δtj) is the exposure at a given pixel. 
>This response curve can be used to determine radiance values in any images acquired by the imaging processing associated with g, not just the images used to recover the response curve.

2. High Dynamic Range Radiance Map Construction
>Once the response curve g is recovered, we can construct a radiance map for each pixel from the response curve (equation 6 in Debevec). In order to reducing noise in the recovered radiance value, we use all the available exposrues for a particular pixel to computer its radiance.

### Tone Mapping
>Global tone mapping: The output image is proportional to the input raised to the power of the inverse of gamma. 

### Color Adjustment
>In order to construct HDR image to be as closer to input image as possible, we adjust the output image average intensity for each channel (B, G, R) to be the same as template image. Typically, we use middle image from image stack as template.

## Result 1
### Original image
![original image](https://github.com/vivianhylee/high-dynamic-range-image/example/output1.png)

<table>
<tr>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/example/sample-00.png" /><br>Nearest-Neighbor</th>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/example/sample-01.png" /><br>Nearest-Neighbor</th>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/example/sample-02.png" /><br>Nearest-Neighbor</th>
</tr>
<tr>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/example/sample-03.png" /><br>Nearest-Neighbor</th>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/example/sample-04.png" /><br>Nearest-Neighbor</th>
<th><img src="https://github.com/vivianhylee/high-dynamic-range-image/example/sample-05.png" /><br>Nearest-Neighbor</th>
</tr>
</table>











