## Image formats 🖼️
- Used to represent images.
- Most commonly used are JPEG, PNG and GIF

![Various image formats](https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Image_formats_by_scope.svg/1024px-Image_formats_by_scope.svg.png)

- Image extension are divided into two families.
  1. Vector ✒️
  2. Raster 🖨️
  
## Vector ✒️
Vector files uses mathematical equations, lines and curves with fixed points on grid to produce an image.
For example: SVG(Scalable Vector Graphics), CGM(Computer Grapics Metafile)<br />
Representation of letter s in SVG

### Advantages ✔️
-  Can be scaled without distortion

### Disadvantages ❌
- Limited in dealing with complex images

```xml
<svg width="400" height="300">
  <g transform="translate(20,20)">
    <g stroke-width="40" fill="none">
      <path d="M100 50 A50 50 0 1 0 50 100 A50 50 0 1 1 0 150" stroke="red"   />
    </g>
    <g stroke-width="8" fill="none">
      <path d="M100 50 A50 50 0 1 0 50 100                   " stroke="blue"  />
      <path d="                    M50 100 A50 50 0 1 1 0 150" stroke="green" />
    </g>
  </g>
</svg>
```
### Raster 
The fundamental strategy underlying the raster data model is the tessellation of a plane, into a two-dimensional array of squares, each called a cell or pixel (from "picture element").

Commonly used raster extensions are JPEG, PNG, etc.

Representation of letter S in raster format

```
[[243 243 243 243 243 243 243 243 243 243]
 [243 243 243 243 243 243 243 243 243 243]
 [243 243 243 243 101 243 243 243 243 243]
 [243 243 243 243 243  19 243 243 243 243]
 [243 243 243   2   2 243 243 243 243 243]
 [243 243 243 243   2   2 243 243 243 243]
 [243 243 243 243 243 139 243 243 243 243]
 [243 243 243  19 243 243 243 243 243 243]
 [243 243 243 243 243 243 243 243 243 243]
 [243 243 243 243 243 243 243 243 243 243]]
```

### Advantages
- Attention to detail
### Disadvantages
- Limited resolution

### Advantages ✔️
- Attention to detail

### Disadvantages ❌
- Limited resolution

### Common raster extensions 🌈
1. ### Jpeg(Joint Photographic Experts Group) 🌐

     -  Commonly used method for lossy compression for digital images
     - 24-bit color and uses lossy compression to compress image
     - Can store metadata such as when was image snapped and camera settings
    - #### Pros ✔️
      - Small file sizes allow for quick transfer and fast access for viewing online
      #### Cons ❌
      - Dealing with very heavily compressed images, the quality will suffer
2. ### BMP 🖼️
    - Developed by Microsoft Windows operating system to maintain the resolution of digital images across different screens and devices.
    - BMP files are lossless and uncompressed.
    - #### Pros ✔️
        - BMP is device-independent and can be stored in multiple devices without losing quality.
    -   #### Cons ❌
        -   Uncompressed BMPs can have larger sizes.

### 3.  GIF(Graphics Interchange Format) 🌌
  - Support up to 8 bits per pixel.
  - Allow images or frames to be combined creating basic animations.
  -  #### pros ✔️
      - Limited color can help them to load faster on web pages.
  - #### Cons ❌
    - Only 8-bit format leads to low image resolution 
### 4.  TIFF( Tag Image File Format) 🏷️
  - Are lossless forms of file compression
  - Enable users to tag up extra image information and data.
  - #### Pros ✔️
      - Retain the original image's detail and color depth and perfect for high-quality professional photos
  - #### Cons ❌
    - Details and resolution lead to TIFFs being quite large files.
### 5. PNG(Portable Network Graphic) 🖌️
  - Supports 24-bit or 32-bit and lossless compression.
      - Can handle graphics with transparent or semi-transparent backgrounds
  - #### Pros ✔️
    - PNG files can store much more detailed images than GIFs.
  - #### Cons ❌
    - PNG files will generally be a lot larger in size than a GIF or JPEG.