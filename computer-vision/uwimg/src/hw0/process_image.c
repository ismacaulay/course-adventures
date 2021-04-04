#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

float get_pixel(image im, int x, int y, int c)
{
    if (x < 0) x = 0;
    if (x >= im.w) x = im.w - 1;

    if (y < 0) y = 0;
    if (y >= im.h) y = im.h - 1;

    if (c < 0) c = 0;
    if (c >= im.c) c = im.c - 1;

    return im.data[(c * im.h * im.w) + (y * im.w) + x];
}

void set_pixel(image im, int x, int y, int c, float v)
{
    if (x < 0 || x >= im.w) return;
    if (y < 0 || y >= im.h) return;
    if (c < 0 || c >= im.c) return;

    im.data[(c * im.h * im.w) + (y * im.w) + x] = v;
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    memcpy(copy.data, im.data, im.w * im.h * im.c * sizeof(float));
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);

    int i, j;
    float g;
    for (i = 0; i < im.w; ++i) {
        for (j = 0; j < im.h; ++j) {
            g = 0.299 * get_pixel(im, i, j, 0) +
                0.587 * get_pixel(im, i, j, 1) +
                0.114 * get_pixel(im, i, j, 2);
            set_pixel(gray, i, j, 0, g);
        }
    }
    return gray;
}

void shift_image(image im, int c, float v)
{
    int i, j;
    float cur;
    for (i = 0; i < im.w; ++i) {
        for (j = 0; j < im.h; ++j) {
            cur = get_pixel(im, i, j, c);
            set_pixel(im, i, j, c, cur + v);
        }
    }
}

void clamp_image(image im)
{
    int i, j, k;
    float cur;
    for (i = 0; i < im.w; ++i) {
        for (j = 0; j < im.h; ++j) {
            for (k = 0; k < im.c; ++k) {
                cur = get_pixel(im, i, j, k);
                if (cur < 0) cur = 0.;
                if (cur > 1) cur = 1.;
                set_pixel(im, i, j, k, cur);
            }
        }
    }
}


// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
    int i, j;
    float r, g, b;
    float h, s, v;
    float c, h_prime;
    for (i = 0; i < im.w; ++i) {
        for (j = 0; j < im.h; ++j) {
            r = get_pixel(im, i, j, 0);
            g = get_pixel(im, i, j, 1);
            b = get_pixel(im, i, j, 2);

            v = three_way_max(r, g, b);

            c = v - three_way_min(r, g, b);
            s = c == 0 ? 0. : c / v;

            if (c == 0) {
                h_prime = 0.;
            } else if (v == r) {
                h_prime = (g - b) / c;
            } else if (v == g) {
                h_prime = ((b - r) / c) + 2.;
            } else if (v == b) {
                h_prime = ((r - g) / c) + 4.;
            }

            h = h_prime / 6.;
            if (h_prime < 0) h += 1.;

            set_pixel(im, i, j, 0, h);
            set_pixel(im, i, j, 1, s);
            set_pixel(im, i, j, 2, v);
        }
    }
}

void hsv_to_rgb(image im)
{
    int i, j;
    float r, g, b;
    float h, s, v;
    float c, h_prime, x, m;
    for (i = 0; i < im.w; ++i) {
        for (j = 0; j < im.h; ++j) {
            h = get_pixel(im, i, j, 0);
            s = get_pixel(im, i, j, 1);
            v = get_pixel(im, i, j, 2);

            c = v == 0 ? 0 : s * v;
            m = v - c;
            h_prime = h * 6.;
            if (0 <= h_prime && h_prime < 1) {
                r = v;
                b = m;
                g = (h_prime * c) + b;
            } else if (1 <= h_prime && h_prime < 2) {
                g = v;
                b = m;
                r = b - ((h_prime - 2) * c);
            } else if (2 <= h_prime && h_prime < 3) {
                g = v;
                r = m;
                b = ((h_prime - 2) * c) + r;
            } else if (3 <= h_prime && h_prime < 4) {
                b = v;
                r = m;
                g = r - ((h_prime - 4) * c);
            } else if (4 <= h_prime && h_prime < 5) {
                b = v;
                g = m;
                r = ((h_prime - 4) * c) + g;
            } else if (5 <= h_prime && h_prime < 6) {
                r = v;
                g = m;
                b = g - ((h_prime - 6) * c);
            }

            set_pixel(im, i, j, 0, r);
            set_pixel(im, i, j, 1, g);
            set_pixel(im, i, j, 2, b);
        }
    }
}

void scale_image(image im, int c, float v)
{
    int i, j;
    float cur;
    for (i = 0; i < im.w; ++i) {
        for (j = 0; j < im.h; ++j) {
            cur = get_pixel(im, i, j, c);
            set_pixel(im, i, j, c, cur * v);
        }
    }
}
