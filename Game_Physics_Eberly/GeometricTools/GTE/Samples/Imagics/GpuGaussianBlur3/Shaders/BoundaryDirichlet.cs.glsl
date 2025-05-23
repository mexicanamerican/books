// David Eberly, Geometric Tools, Redmond WA 98052
// Copyright (c) 1998-2025
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
// https://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
// Version: 6.0.2022.01.03

layout(r32f) uniform readonly image2D inImage;
layout(r32f) uniform readonly image2D inMask;
layout(r32f) uniform writeonly image2D outImage;

layout(local_size_x = NUM_X_THREADS, local_size_y = NUM_Y_THREADS, local_size_z = 1) in;
void main()
{
    ivec2 t = ivec2(gl_GlobalInvocationID.xy);
    float result = imageLoad(inMask, t).r * imageLoad(inImage, t).r;
    imageStore(outImage, t, vec4(result, 0, 0, 0));
}
