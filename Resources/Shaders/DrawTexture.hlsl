//------------------------------------------------------------------------------------------
//	Constant Buffer
//------------------------------------------------------------------------------------------
cbuffer CONSTANT_BUFFER :register(b0)
{
    matrix WVP            : packoffset(c0);
    float  Time           : packoffset(c4);
    float2 Mouse          : packoffset(c5);
    float2 Roughness      : packoffset(c6);
    float2 MicroRoughness : packoffset(c7);
    float Variation       : packoffset(c8);
    float Density         : packoffset(c9);
};

//------------------------------------------------------------------------------------------
//	Structure
//------------------------------------------------------------------------------------------
struct VS_OUTPUT
{
    float4 Pos : SV_POSITION;
};

//------------------------------------------------------------------------------------------
//  Vertex Shader
//------------------------------------------------------------------------------------------
VS_OUTPUT VS(float3 Pos : POSITION)
{
    VS_OUTPUT output = (VS_OUTPUT)0;
    output.Pos = mul(float4(Pos, 1.0), WVP);

    return output;
}

//------------------------------------------------------------------------------------------
//	Constant values
//------------------------------------------------------------------------------------------
static float2 RESOLUTION = { 512, 512 };

//------------------------------------------------------------------------------------------
//  Functions
//------------------------------------------------------------------------------------------

//=========================================================================
//  procedural distance fields
//  http://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
//=========================================================================
float sdBox(float3 p, float3 b)
{
    float3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}
float sdEllipsoid(in float3 p, in float3 r)
{
    return (length(p / r) - 1.0) * min(min(r.x, r.y), r.z);
}
float sdRoundBox(float3 p, float3 b, float r)
{
    float3 d = abs(p) - b;
    return length(max(d, 0.0)) - r + min(max(d.x, max(d.y, d.z)), 0.0);
}
float sdCylinder(float3 p, float2 h)
{
    float2 d = abs(float2(length(p.xz), p.y)) - h;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}
float length2(float2 p)
{
    return sqrt(p.x * p.x + p.y * p.y);
}
float length8(float2 p)
{
    p = p * p;
    p = p * p;
    p = p * p;
    return pow(p.x + p.y, 1.0 / 8.0);
}
float sdTorus82(float3 p, float2 t)
{
    float2 q = float2(length2(p.xz) - t.x, p.y);
    return length8(q) - t.y;
}
//Substraction
float opS(float d1, float d2)
{
    return max(-d2, d1);
}
//Union
float2 opU(float2 d1, float2 d2)
{
    return (d1.x < d2.x) ? d1 : d2;
}
// polynomial smooth min (k = 0.1); http://iquilezles.org/www/articles/smin/smin.htm
float2 smin(float2 a, float2 b, float k)
{
    float h = clamp(0.5 + 0.5 * (b.x - a.x) / k, 0.0, 1.0);
    return lerp(b, a, h) - float2(k * h * (1.0 - h), 0.0);
}

//=========================================================================
// scene
//=========================================================================
float2 wheel(in float3 pos)
{
    float2 res = float2(sdTorus82(pos.yxz, float2(0.65, 0.15)), 2.0);
    res = opU(res, float2(sdCylinder(pos.yxz, float2(0.65, 0.08)), 3.0));
    return res;
}
float2 fenders(float2 res, in float3 pos, float material)
{
    res = smin(res, float2(sdCylinder(pos.yxz, float2(1.0, 1.65)), material), 0.4);
    res.x = opS(res.x, sdCylinder(pos.yxz, float2(0.9, 1.9)));
    res.x = opS(res.x, sdBox(pos - float3(0, -1.1, 0), float3(2.0, 1.0, 3.0)));
    return res;
}
float2 cabin(in float3 pos, float material)
{
    float2 res = float2(sdRoundBox(pos - float3(0, 1.5, 0.1), float3(0.9, 0.1, 2.4), 0.6), material);
    res = smin(res, float2(sdEllipsoid(pos - float3(0, 1.5, -3.0), float3(1.3, 0.7, 0.9)), material), 1.0);
    res = smin(res, float2(sdEllipsoid(pos - float3(0, 1.6, 3.3), float3(1.2, 0.8, 0.5)), material), 1.0);
    res = smin(res, float2(sdRoundBox(pos - float3(0, 2.0, 0.3), float3(0.6, 0.3, 1.0), 0.6), material), 0.7);
    
    res = fenders(res, pos - float3(0, 0.9, -2.5), material);
    res = fenders(res, pos - float3(0, 0.9, 2.5), material);
    
    res.x = opS(res.x, sdRoundBox(pos - float3(0, 2.6, 0.3), float3(0.6, 0.0, 3.0), 0.25));
    res.x = opS(res.x, sdRoundBox(pos - float3(0, 2.6, -0.3), float3(3.0, 0.1, 0.4), 0.15));
    res.x = opS(res.x, sdRoundBox(pos - float3(0, 2.6, 0.8), float3(3.0, 0.1, 0.2), 0.15));
    res.x = opS(res.x, sdRoundBox(pos - float3(0, 2.0, 0.3), float3(0.6, 0.3, 1.0), 0.5));
    
    return res;
}
float2 car(in float3 pos, float material)
{
    float2 res = float2(sdBox(pos - float3(0, 0.8, 0), float3(1.3, 0.3, 3.0)), 2.0);
    res = opU(res, wheel(pos - float3(1.5, 0.8, 2.5)));
    res = opU(res, wheel(pos - float3(-1.5, 0.8, 2.5)));
    res = opU(res, wheel(pos - float3(1.5, 0.8, -2.5)));
    res = opU(res, wheel(pos - float3(-1.5, 0.8, -2.5)));
    
    res = opU(cabin(pos, material), res);
    return res;
}
// heightfield
float mapLandscape(float2 a)
{
    float h = 0.0;
    h += pow(sin(2.0 + a.x / 37.0) * sin(0.5 + a.y / 31.0), 2.0);
    h += 0.5 * pow(sin(a.x / 11.0) * sin(a.y / 13.0), 2.0);
    h += 0.2 * sin(1.1 + a.x / 3.1) * sin(0.3 + a.y / 3.7);
    h *= min(length(a) / 10.0, 5.0);
    h += 0.1 * sin(0.9 + a.x / 1.7);
    h += 0.1 * sin(0.4 + a.y / 1.4);
    h += 0.05 * sin(1.1 + a.y / 0.9);
    h += 15.0 * (1.0 - cos(a.x / 51.0));
    return h;
}
float2 map(in float3 pos)
{
    float2 res = car(pos - float3(0, 0, 0), 4.3);
    res = opU(car(pos - float3(9, 0, 5), 5.995), res);
    return res;
}

//=========================================================================
// ray casting
//=========================================================================
float2 castRayLandscape(in float3 ro, in float3 rd)
{
    float delt = 0.1;
    const float mint = 0.5;
    const float maxt = 90.0;
    float lh = 0.0;
    float ly = 0.0;
    float t = mint;
    for (int i = 0; i < 200; ++i)
    {
        //コメントアウトするとうまくいく。よーわからん。
        //if (t < maxt);
        //else break;
        
        float3 p = ro + rd * t;
        float h = mapLandscape(p.xz);
        if (p.y < h && t < maxt)
        {
            // interpolate the intersection distance
            return float2(t - delt + delt * (lh - ly) / (p.y - ly - h + lh), 1.99);
        }
        // allow the error to be proportinal to the distance
        delt = max(0.1 * t, delt);
        lh = h;
        ly = p.y;
        
        t += delt;
    }
    return float2(maxt, -1.0);
}
float3 calcNormalLandscape(in float3 pos)
{
    float3 eps = float3(0.01, 0.0, 0.0);
    float3 a = eps.xyz;
    a.y = mapLandscape(pos.xz + a.xz) - mapLandscape(pos.xz - a.xz);
    a.xz *= 2.0;
    float3 b = eps.zyx;
    b.y = mapLandscape(pos.xz + b.xz) - mapLandscape(pos.xz - b.xz);
    b.xz *= 2.0;
    return normalize(cross(b, a));
}
float2 castRay(in float3 ro, in float3 rd)
{
    float tmin = 1.0;
    float tmax = 60.0;
    
    float precis = 0.01;
    float t = tmin;
    float m = -1.0;
    float2 res;
    for (int i = 0; i < 32; i++)
    {
        res = map(ro + rd * t);
        if (res.x < precis || t > tmax)
        {
            break;
        }
        t += res.x;
        m = res.y;
    }

    if (t > tmax) // || res.x > 100.0 * precis
    {
        m = -1.0;
    }
    res = float2(t, m);
    
    float2 resLandscape = castRayLandscape(ro, rd);
    if (resLandscape.y >= 0.0 && (m < 0.0 || resLandscape.x < t))
    {
        res = resLandscape;
    }
        
    return res;
}
float3 calcNormal(in float3 pos)
{
    float3 eps = float3(0.01, 0.0, 0.0);
    float3 nor = float3(
      map(pos + eps.xyy).x - map(pos - eps.xyy).x,
      map(pos + eps.yxy).x - map(pos - eps.yxy).x,
      map(pos + eps.yyx).x - map(pos - eps.yyx).x);
    return normalize(nor);
}
float softshadow(in float3 ro, in float3 rd, in float mint, in float tmax)
{
    float res = 1.0;
    float t = mint;
    for (int i = 0; i < 24; i++)
    {
        float h = map(ro + rd * t).x;
        res = min(res, 8.0 * h / t);
        t += clamp(h, 0.04, 1.0);
        if (h < 0.001 || t > tmax)
        {
            break;
        }
    }
    return clamp(res, 0.0, 1.0);
}

//=========================================================================
// glints
//=========================================================================
//mod(glsl)とfmod(hlsl)では負の値の時返す値が異なってしまうからmodは自作
float mod(float x, float y)
{
    return x - y * floor(x / y);
}

//hash
float hash(float n)
{
    return frac(sin(mod(n, 3.14)) * 753.5453123);
}
float2 hash2(float n)
{
    return float2(hash(n), hash(1.1 + n));
}

//math
float compMax(float2 v)
{
    return max(v.x, v.y);
}
float maxNrm(float2 v)
{
    return compMax(abs(v));
}
float2x2 inverse2(float2x2 m)
{
    return float2x2(m[1][1], -m[0][1], -m[1][0], m[0][0]) / (m[0][0] * m[1][1] - m[0][1] * m[1][0]);
}
float erfinv(float x)
{
    float w, p;
    w = -log((1.0 - x) * (1.0 + x));
    if (w < 5.000000)
    {
        w = w - 2.500000;
        p = 2.81022636e-08;
        p = 3.43273939e-07 + p * w;
        p = -3.5233877e-06 + p * w;
        p = -4.39150654e-06 + p * w;
        p = 0.00021858087 + p * w;
        p = -0.00125372503 + p * w;
        p = -0.00417768164 + p * w;
        p = 0.246640727 + p * w;
        p = 1.50140941 + p * w;
    }
    else
    {
        w = sqrt(w) - 3.000000;
        p = -0.000200214257;
        p = 0.000100950558 + p * w;
        p = 0.00134934322 + p * w;
        p = -0.00367342844 + p * w;
        p = 0.00573950773 + p * w;
        p = -0.0076224613 + p * w;
        p = 0.00943887047 + p * w;
        p = 1.00167406 + p * w;
        p = 2.83297682 + p * w;
    }
    return p * x;
}

// ray differentials
void calcDpDxy(in float3 ro, in float3 rd, in float3 rdx, in float3 rdy, in float t, in float3 nor,
                out float3 dpdx, out float3 dpdy)
{
    dpdx = 2.0 * t * (rdx * dot(rd, nor) / dot(rdx, nor) - rd) * sign(dot(rd, rdx));
    dpdy = 2.0 * t * (rdy * dot(rd, nor) / dot(rdy, nor) - rd) * sign(dot(rd, rdy));
}

// some microfacet BSDF geometry factors
// (divided by NoL * NoV b/c cancelled out w/ microfacet BSDF)
float geometryFactor(float NoL, float NoV, float2 roughness)
{
    float a2 = roughness.x * roughness.y;
    NoL = abs(NoL);
    NoV = abs(NoV);

    float G_V = NoV + sqrt((NoV - NoV * a2) * NoV + a2);
    float G_L = NoL + sqrt((NoL - NoL * a2) * NoL + a2);
    return 1.0 / (G_V * G_L);
}

//----------------------------------------------------------------------
// ugly inefficient WebGL implementation of simple bit shifts for
// multilevel coherent grid indices. See comment in multilevelGridIdx.
int multilevelGridIdx1(inout int idx)
{
    for (int i = 0; i < 32; ++i)
    {
        if (idx / 2 == (idx + 1) / 2)
            idx /= 2;
        else
            break;
    }
    return idx;
}
int2 multilevelGridIdx(int2 idx)
{
//  return idx >> findLSB(idx); // findLSB not supported by Shadertoy WebGL version
    return int2(multilevelGridIdx1(idx.x), multilevelGridIdx1(idx.y));
}

//----------------------------------------------------------------------
// stable binomial 'random' numbers: interpolate between result for
// two closest binomial distributions where log_{.9}(p_i) integers
float binomial_interp(float u, float N, float p)
{
    if (p >= 1.0)
        return N;
    else if (p <= 1e-10)
        return 0.0;

    // convert to distribution on ints while retaining expected value
    float cN = ceil(N);
    int iN = int(cN);
    p = p * (N / cN);
    N = cN;

    // round p to nearest powers of .9 (more stability)
    float pQ = 0.9;
    float pQef = log2(p) / log2(pQ);
    float p2 = exp2(floor(pQef) * log2(pQ));
    float p1 = p2 * pQ;
    float2 ps = float2(p1, p2);

    // compute the two corresponding binomials in parallel
    float2 pm = pow(1.0 - ps, float2(N, N));
    float2 cp = pm;
    float2 r = float2(N, N);

    float i = 0.0;
    for (int ii = 0; ii <= N; ++ii)
    {
        if (u < cp.x)
            r.x = min(i, r.x);
        if (u < cp.y)
        {
            r.y = i;
            break;
        }
        // fast path
        if (ii > 16)
        {
            float C = 1.0 / (1.0 - pow(p, N - i - 1.));
            float2 U = (u - cp) / (1.0 - cp);
            float2 A = (i + 1.0 + log2(1.0 - U / C) / log2(p));
            r = min(A, r);
            break;
        }

        i += 1.0;
        pm /= 1. - ps;
        pm *= (N + 1.0 - i) / i;
        pm *= ps;
        cp += pm;
    }

    // interpolate between the two binomials according to log p (akin to mip interpolation)
    return lerp(r.y, r.x, frac(pQef));
}
// resort to gaussian distribution for larger N*p
float approx_binomial(float u, float N, float p)
{
    if (p * N > 5.0)
    {
        float e = N * p;
        float v = N * p * max(1.0 - p, 0.0);
        float std = sqrt(v);
        float k = e + erfinv(lerp(-0.999999, 0.999999, u)) * std;
        return min(max(k, 0.0), N);
    }
    else
        return binomial_interp(u, N, p);
}

//----------------------------------------------------------------------

float3 glints(float2 texCO, float2 duvdx, float2 duvdy, float3x3 ctf
  , float3 lig, float3 nor, float3 view
  , float2 roughness, float2 microRoughness, float searchConeAngle, float variation, float dynamicRange, float density)
{
    float3 col = float3(0.0, 0.0, 0.0);

    // Section 4.2
    // Compute pixel footprint in texture space, step size w.r.t. anisotropy of the footprint
    float2x2 uvToPx = inverse2(float2x2(duvdx, duvdy));
    float2 uvPP = 1.0 / float2(maxNrm(uvToPx[0]), maxNrm(uvToPx[1]));

    // Section 4.3
    // material
    float2 mesoRoughness = sqrt(max(roughness * roughness - microRoughness * microRoughness, float2(1.e-12, 1.e-12))); // optimizer fail, max 0 removed

    // Anisotropic compression of the grid (gamma_m)
    float2 texAnisotropy = float2(min(mesoRoughness.x / mesoRoughness.y, 1.0),
                                    min(mesoRoughness.y / mesoRoughness.x, 1.0));

    // Compute half floattor (w.r.t. dir light)
    float3 hvW = normalize(lig + view);
    float3 hv = normalize(mul(ctf, hvW));
    float2 h = hv.xy / hv.z;
    float2 h2 = 0.75 * hv.xy / (hv.z + 1.0);
    // Anisotropic compression of the slope-domain grid
    h2 *= texAnisotropy;

    // Compute the Gaussian probability of encountering a glint within a given finite cone
    float2 hppRScaled = h / roughness;
    float pmf = (microRoughness.x * microRoughness.y) / (roughness.x * roughness.y)
                    * exp(-dot(hppRScaled, hppRScaled)); // planeplane h
    pmf /= hv.z * hv.z * hv.z * hv.z; // projected h
    //pmf /= dot(lig, nor) * dot(view, nor); // projected area, cancelled out by parts of G, ...
    float pmfToBRDF = 1.0 / (3.14159 * microRoughness.x * microRoughness.y);
    pmfToBRDF /= 4.0; // solid angle o
    pmfToBRDF *= geometryFactor(dot(lig, nor), dot(view, nor), roughness);
    // phenomenological: larger cones flatten distribution
    float searchAreaProj = searchConeAngle * searchConeAngle / (4.0 * dot(lig, hvW) * hv.z); // * PI
    pmf = lerp(pmf, 1.0, clamp(searchAreaProj, 0.0, 1.0)); // searchAreaProj / PI
    pmf = min(pmf, 1.0);
    
    // noise coordinate (decorrelate interleaved grid)
    texCO += float2(100.0, 100.0);
    // apply anisotropy _after_ footprint estimation
    texCO *= texAnisotropy;

    // Compute AABB of pixel in texture space
    float2 uvAACB = max(abs(duvdx), abs(duvdy)) * texAnisotropy; // border center box
    float2 uvb = texCO - 0.5 * uvAACB;
    float2 uve = texCO + 0.5 * uvAACB;

    float2 uvLongAxis = uvAACB.x > uvAACB.y ? float2(1.0, 0.0) : float2(0.0, 1.0);
    float2 uvShortAxis = 1.0 - uvLongAxis;

    // Compute skew correction to snap axis-aligned line sampling back to longer anisotropic pixel axis in texture space
    float2 skewCorr2 = -(mul(uvLongAxis, uvToPx)) / (mul(uvShortAxis, uvToPx));
    float skewCorr = abs((mul(uvShortAxis, uvToPx)).x) > abs((mul(uvShortAxis, uvToPx)).y) ? skewCorr2.x : skewCorr2.y;
    skewCorr *= dot(texAnisotropy, uvShortAxis) / dot(texAnisotropy, uvLongAxis);

    float isoUVPP = dot(uvPP, uvShortAxis);
    // limit anisotropy
    isoUVPP = max(isoUVPP, dot(uvAACB, uvLongAxis) / 16.0);     //Ka=16

     // Two virtual grid mips: current and next
    float fracMip = log2(isoUVPP);
    float lowerMip = floor(fracMip);
    float uvPerLowerC = exp2(lowerMip);

    // Current mip level and cell size
    float uvPC = uvPerLowerC;
    float mip = lowerMip;

    int iter = 0;
    int iterThreshold = 60;

    for (int i = 0; i < 2; ++i)
    {
        float mipWeight = 1.0 - abs(mip - fracMip);

        float2 uvbg = min(uvb + 0.5 * uvPC, texCO);
        float2 uveg = max(uve - 0.5 * uvPC, texCO);

        // Snapped uvs of the cell centers
        float2 uvbi = floor(uvbg / uvPC);
        float2 uvbs = uvbi * uvPC;
        float2 uveo = uveg + uvPC - uvbs;

        // Resulting compositing values for a current layer
        float weight = 0.0;
        float3 reflection = float3(0.0, 0.0, 0.0);

        // March along the long axis
        float2 uvo = float2(0.0, 0.0);
        float2 uv = uvbs;
        float2 uvio = float2(0.0, 0.0);
        float2 uvi = uvbi;
        for (int iter1 = 0; iter1 < 18; ++iter1) // horrible WebGL-compatible static for loop
        {
            // for cond:
            if (dot(uvo, uvLongAxis) < dot(uveo, uvLongAxis) && iter < iterThreshold);
            else
                break;

            // Snap samples to long anisotropic pixel axis
            float uvShortCenter = dot(texCO, uvShortAxis) + skewCorr * dot(uv - texCO, uvLongAxis);

            // Snapped uvs of the cell center
            uvi += (floor(uvShortCenter / uvPC) - dot(uvi, uvShortAxis)) * uvShortAxis;
            uv = uvi * uvPC;
            float uvShortEnd = uvShortCenter + uvPC;

            float2 uvb2 = uvbg * uvLongAxis + uvShortCenter * uvShortAxis;
            float2 uve2 = uveg * uvLongAxis + uvShortCenter * uvShortAxis;

            // March along the shorter axis
            for (int iter2 = 0; iter2 < 4; ++iter2) // horrible WebGL-compatible static for loop
            {
                // for cond:
                if (dot(uv, uvShortAxis) < uvShortEnd && iter < iterThreshold);
                else
                    break;

                // Compute interleaved cell index
                int2 cellIdx = int2(uvi + float2(0.5, 0.5));
                cellIdx = multilevelGridIdx(cellIdx);

                // Randomize a glint based on a texture-space id of current grid cell
                float2 u2 = hash2(float((cellIdx.x + 1549 * cellIdx.y)));
                // Compute index of the cone
                float2 hg = h2 / (microRoughness + searchConeAngle);
                float2 hs = floor(hg + u2) + u2 * 533.0; // discrete cone index in paraboloid hv grid
                int2 coneIdx = int2(hs);

                // Randomize glint sizes within this layer
                float var_u = hash(float((cellIdx.x + cellIdx.y * 763 + coneIdx.x + coneIdx.y * 577)));
                float mls = 1. + variation * erfinv(lerp(-0.999, 0.999, var_u));
                if (mls <= 0.0)
                    mls = frac(mls) / (1. - mls);
                mls = max(mls, 1.e-12);

                // Bilinear interpolation using coverage made by areas of two rects
                float2 mino = max(1.0 - max((uvb2 - uv) / uvPC, 0.0), 0.0);
                float2 maxo = max(1.0 - max((uv - uve2) / uvPC, 0.0), 0.0);
                float2 multo = mino * maxo;
                float coverageWeight = multo.x * multo.y;

                float cellArea = uvPC * uvPC;
                // Expected number of glints 
                float eN = density * cellArea;
                float sN = max(eN * mls, min(1.0, eN));
                eN = eN * mls;

                // Sample actually found number of glints
                float u = hash(float(coneIdx.x + coneIdx.y * 697));
                float lN = approx_binomial(u, sN, pmf);

                // Ratio of glinting vs. expected number of microfacets
                float ratio = lN / eN;
                
                // limit dynamic range (snow more or less unlimited)
                ratio = min(ratio, dynamicRange * pmf);
                
                // convert to reflectance
                ratio *= pmfToBRDF;

                // Accumulate results
                reflection += coverageWeight * ratio;
                //reflection += float3(u, u, u);    // Grid for Debug
                weight += coverageWeight;

                // for incr:
                uv += uvPC * uvShortAxis;
                uvi += uvShortAxis;
                ++iter;
            }

            // for incr:
            uvo += uvPC * uvLongAxis;
            uv = uvbs + uvo;
            uvio += uvLongAxis;
            uvi = uvbi + uvio;
        }

        reflection = reflection / weight;

        // Compositing of two layers
        col += mipWeight * reflection;

        // for incr:
        uvPC *= 2.0;
        mip += 1.0;
    }

    return col;
}


//=========================================================================
// render
//=========================================================================
float3 render(in float3 ro, in float3 rd, in float3 rdx, in float3 rdy)
{
    // sun and sky
    float3 lightDir = normalize(float3(0.6, .9, 0.5));
    float3 lightPower = float3(9.0, 9.0, 9.0);
    float3 sky = float3(0.7, 0.9, 1.0) + 1.0 + rd.y * 0.8;
    float3 col = sky * lightPower;

    // ray cast
    float2 res = castRay(ro, rd);
    float t = res.x;
    float m = res.y;

    // shade hit
    if (m > -0.5)
    {
        // hit information
        float3 pos = ro + t * rd;
        float3 normal = (m < 2.0) ? calcNormalLandscape(pos) : calcNormal( pos );   //  if m>2, it is car object 

        float3x3 texProjFrame = float3x3(float3(1, 0, 0), float3(0, 0, 1), float3(0, 1, 0));
        if (abs(normal.x) > abs(normal.y) && abs(normal.x) > abs(normal.z))
        {
            texProjFrame = float3x3(float3(0, 0, 1), float3(0, 1, 0), float3(1, 0, 0));
        }
        else if (abs(normal.z) > abs(normal.x) && abs(normal.z) > abs(normal.y))
        {
            texProjFrame = float3x3(float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1));
        }  
        float3 bitang = normalize(cross(normal, texProjFrame[0]));
        float3 tang = cross(bitang, normal);
        float3x3 ctf = float3x3(tang, bitang, normal);
        
        // texturing
        float3 dposdx, dposdy;
        calcDpDxy(ro, rd, rdx, rdy, t, normal, dposdx, dposdy);
        // planar projections
        float texScaling = 1.0;
        float2 texCO = texScaling * (mul(texProjFrame, pos)).xy;
        float2 duvdx = texScaling * (mul(texProjFrame, dposdx)).xy;
        float2 duvdy = texScaling * (mul(texProjFrame, dposdy)).xy;
        
        // information
        float occ = softshadow(pos, lightDir, 0.02, 25.0);
        float amb = clamp(0.5 + 0.5 * normal.y, 0.0, 1.0);
        float dif = clamp(dot(normal, lightDir), 0.0, 1.0);
        float fre = 1.0 - pow(1.0 - dif, 2.5);
        float dfr = (1.0 - pow(1.0 - clamp(dot(normal, -rd), 0.0, 1.0), 2.5)) * fre;
        float specularity = frac(m);
        
        // configure multiscale material (snow parameters)
        float2 roughness = float2(0.6, 0.6);
        float2 microRoughness = roughness * 0.024;
        float searchConeAngle = 0.01;
        float variation = 100.0;
        float dynamicRange = 50000.0;
        float density = 5.e8;
    
        // snow
        if (floor(m) == 1.0)
        {
            col = lerp(float3(0.1, 0.2, 0.5), float3(0.95, 0.8, 0.75), (1.0 - abs(rd.y)) * dif);
        }
        // wheels (not using multiscale, specularity = 0)
        else if (floor(m) == 2.0)
        {
            col = float3(0,0,0);
        }
        else if (floor(m) == 3.)
        {
            col = float3(0.7, 0.7, 0.7);
        }
        // car 1 (anisotropic)
        else if (floor(m) == 4.0)
        {
            col = float3(0.02, 0.2, 0.04);
            col = lerp(col, sky, 0.15 * pow(1.0 - dfr, 2.0));
            roughness = Roughness;
            microRoughness = MicroRoughness;
            variation = Variation;
            dynamicRange = 10.0; // max 10x more microdetails than expected
            density = Density;
            specularity *= dfr; // layered material (translucency)
        }
        // car 2 (isotropic)
        else if (floor(m) == 5.0)
        {
            col = float3(0.5, 0.025, 0.025);
            col = lerp(col, sky, 0.15 * pow(1.0 - dfr, 2.0));
            roughness = float2(0.07, 0.07);
            microRoughness = roughness * 0.5;
            variation = 0.1;
            dynamicRange = 5.0; // max 5x more microdetails than expected
            density = 1.e6;
            specularity *= dfr; // layered material (translucency)
        }
        
        // standard diffuse lighting
        col *= lightPower * lerp(0.02, 1.0, occ * dif);
        
        // multiscale specular lighting
        if (specularity > 0.0 && dif > 0.0 && dot(-rd, normal) > 0.0)
            col += specularity * glints(texCO, duvdx, duvdy, ctf, lightDir, normal, -rd, roughness, microRoughness, searchConeAngle, variation, dynamicRange, density)
                    * lightPower * lerp(0.05, 1.0, occ);
    }
    return col;
}

float3x3 setCamera(in float3 ro, in float3 ta, float cr)
{
    float3 cw = normalize(ta - ro);
    float3 cp = float3(sin(cr), cos(cr), 0.0);
    float3 cu = normalize(cross(cw, cp));
    float3 cv = normalize(cross(cu, cw));
    return float3x3(cu, cv, cw);
}

//------------------------------------------------------------------------------------------
//  Pixel Shader
//------------------------------------------------------------------------------------------
float4 PS(VS_OUTPUT input) : SV_Target
{
    //　-1~1のuv座標をつくる
    float4 Pos = input.Pos;
    float2 uv = -1.0 + 2.0 * Pos.xy / RESOLUTION.xy;
    uv.x *= RESOLUTION.x / RESOLUTION.y;
    uv.y = -uv.y;
    float2 mouse = Mouse.xy / RESOLUTION.xy;
    
    // camera
    float ds = 1.5 + sin(Time / 2.0);
    float3 eyePos = float3(-0.5 + ds * 8.5 * cos(0.1 * Time + 6.0 * mouse.x)
                        , 10.0 - 9.5 * mouse.y
                        , 0.5 + ds * 8.5 * sin(0.1 * Time + 6.0 * mouse.x));
    eyePos.y /= 1. + .01 * dot(eyePos.xz, eyePos.xz);
    eyePos.y += mapLandscape(eyePos.xz);

    float3 lookAt = float3(-0.5, -0.4, 0.5);
  
    // camera-to-world transformation
    float3x3 MV = setCamera(eyePos, lookAt, 0.0);
    
    // ray direction
    float3 rd = mul(normalize(float3(uv.xy, 2.0)), MV);
    float2 rds = -sign(uv + 0.001);
    float3 rdx = mul(rds.x * normalize(float3(uv.xy + rds.x * float2(1.0 / RESOLUTION.y, 0), 2.0)), MV);
    float3 rdy = mul(rds.y * normalize(float3(uv.xy + rds.y * float2(0, 1.0 / RESOLUTION.y), 2.0)), MV);

    // render 
    float3 col = render(eyePos, rd, rdx, rdy);

    // tonemap, gamma
    col *= 1.0 / (max(max(col.r, col.g), col.b) + 1.0);
    col = pow(col, 0.4545);

    return float4(col, 1);
}