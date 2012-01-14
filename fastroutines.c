#include "stdio.h"
#include "math.h"
#include "string.h"
#include "gd.h"

typedef struct {
  float x, y, z, w;
} __attribute__ ((aligned (16))) vec3f;

typedef struct {
  float x, y;
} __attribute__ ((aligned (4))) vec2f;

struct Ray {
  vec3f pos, dir;
};

struct Result {
  float distance;
  vec3f normal, col, emissive_col;
  vec2f texcoord; void *texinfo;
  int data;
  int success;
};

struct VMState {
  int resid, rayid;
  int stream_len; void** stream_ptr;
  
  int state;
  float state2;
  int burnInCounter;
  
  vec3f rayCache;
  int cached, cachedBack;
};

typedef float v4sf __attribute__ ((vector_size (16)));
typedef int v4si __attribute__ ((vector_size (16)));

#define ALIGNED __attribute__ ((force_align_arg_pointer))
#define FOUR(x){x,x,x,x}
#define X(vec) __builtin_ia32_vec_ext_v4sf ((vec), 0)
#define Y(vec) __builtin_ia32_vec_ext_v4sf ((vec), 1)
#define Z(vec) __builtin_ia32_vec_ext_v4sf ((vec), 2)
#define W(vec) __builtin_ia32_vec_ext_v4sf ((vec), 3)
#define IX(vec) __builtin_ia32_vec_ext_v4si ((vec), 0)
#define IY(vec) __builtin_ia32_vec_ext_v4si ((vec), 1)
#define IZ(vec) __builtin_ia32_vec_ext_v4si ((vec), 2)
#define IW(vec) __builtin_ia32_vec_ext_v4si ((vec), 3)

float fov;

void ALIGNED coordsf_to_ray(int dw, int dh, float x, float y, struct Ray *rayp) {
  float ratio = dw * 1.0f / dh;
  v4sf v = (v4sf) {ratio * fov * (x / (dw / 2.0) - 1.0), fov * (1.0 - y / (dh / 2.0)), 1.0, 0.0};
  v4sf res = v;
  v *= v;
  float f = 1.0f / sqrtf(*(float*) &v + *((float*) &v + 1) + *((float*) &v + 2));
  
  res *= (v4sf) FOUR(f);
  *(v4sf*) &rayp->pos = (v4sf) {0,2,0,0};
  *(v4sf*) &rayp->dir = (v4sf) res;
}

void ALIGNED coords_to_ray(int dw, int dh, int x, int y, struct Ray *rayp) {
  coordsf_to_ray(dw, dh, (float) x, (float) y, rayp);
}

void ALIGNED ray_to_coordsf(int dw, int dh, struct Ray *rayp, float *xp, float *yp) {
  float ratio = dw * 1.0f / dh;
  v4sf dir = *(v4sf*) &rayp->dir;
  dir /= (v4sf) FOUR(Z(dir)); // denormalize
  *xp = (1.0f + (X(dir) / (ratio * fov))) * (dw / 2.0);
  *yp = (1.0f - (Y(dir) / fov)) * (dh / 2.0);
}

void ALIGNED ray_to_coords(int dw, int dh, struct Ray *rayp, int *xp, int *yp) {
  float x, y;
  ray_to_coordsf(dw, dh, rayp, &x, &y);
  *xp = (int) (x + 0.5);
  *yp = (int) (y + 0.5);
}

void ALIGNED fastsetup(struct Ray **rayplanes, int from, int to, int dw, int dh, int stepsize, struct VMState *state) {
  // float ratio = dw * 1.0f / dh;
  struct Ray *rayplane = rayplanes[0];
  int i = 0;
  for (int k = from; k < to; ++k) {
    if (++i != stepsize) continue;
    i = 0;
    state->resid = 0;
    state->rayid = 1;
    int x = k % dw, y = k / dw;
    coords_to_ray(dw, dh, x, y, rayplane);
    state ++; rayplane ++;
  }
}

#define SUM(vec) \
  ((vec)\
     + __builtin_ia32_shufps((vec), (vec), 0x55)\
     + __builtin_ia32_shufps((vec), (vec), 0xaa)\
  )
// benched as fastest
// TODO: rebench
// #define XSUM(vec) X(SUM(vec))
// Why is this slower? WE MAY NEVER KNOW.
#define SUMX(vec) (X(vec) + X(__builtin_ia32_shufps((vec), (vec), 0x55)) + X(__builtin_ia32_shufps((vec), (vec), 0xaa)))
/*#define XSUM(vec) (\
    __builtin_ia32_vec_ext_v4sf ((vec), 0) \
  + __builtin_ia32_vec_ext_v4sf ((vec), 1) \
  + __builtin_ia32_vec_ext_v4sf ((vec), 2))*/
#define XSUM(vec) __builtin_ia32_vec_ext_v4sf(\
  __builtin_ia32_shufps((vec), (vec), 0xaa)\
+ __builtin_ia32_haddps((vec), (vec))\
, 0)
// #define XSUM(vec) (*(float*) &(vec) + *((float*) &(vec) + 1) + *((float*) &(vec) + 2))
#define XYZ(v) (v).x, (v).y, (v).z
#define V4SF(v) (*(v4sf*) &(v))
#define V4SI(v) (*(v4si*) &(v))

#define LIKELY(X) (__builtin_expect((X), 1))
#define UNLIKELY(X) (__builtin_expect((X), 0))

// IMPORTANT: use -mstackrealign!
void ALIGNED fast_sphere_process(
  struct Ray **rayplanes, struct Result **resplanes,
  struct VMState *states, int numstates,
  vec3f center, float rsq,
  void* self
) {
// #define PREFETCH_HARD(X, READ, LOCALITY) \
//   __builtin_prefetch(X, READ, LOCALITY); \
//   __asm__ volatile ("" : : : "memory"); // force break
#define PREFETCH_HARD(X, READ, LOCALITY) \
  __builtin_prefetch(X, READ, LOCALITY); // volatile makes no difference
  for (int i = 0; i < numstates; ++i) {
    struct VMState* sp = states++;
    PREFETCH_HARD(sp, 1, 3);
    
    if (sp->stream_ptr[0] != self) continue;
    struct Ray* RAY = rayplanes[sp->rayid - 1] + i;
    
    sp->stream_ptr ++; sp->stream_len --;
    struct Result *res = resplanes[sp->resid ++] + i;
    PREFETCH_HARD(RAY, 0, 0);
    
    // pos = ray.pos - center; pretranslate so we can pretend we're a sphere around (0, 0, 0)
    v4sf pos = V4SF(RAY->pos) - V4SF(center);
    v4sf dir = V4SF(RAY->dir);
    
    // algo 1
    float k;
    {
      v4sf dp = dir * pos, dd = dir * dir;
      k = -XSUM(dp) / XSUM(dd);
    }
    v4sf p = pos + dir * (v4sf) FOUR(k);
    p *= p;
    float ps = XSUM(p);
    if (ps > rsq) { res->success = 0; continue; }
    
    float sq = sqrtf (rsq - ps);
    float k1 = k + sq, k2 = k - sq;
    // algo 2
    /*
    // prod = 2 * pos * dir
    v4sf prod = (v4sf) FOUR(2) * pos * dir;
    // p = sum(2 * pos * dir)
    float p = XSUM(prod);

    pos *= pos;
    float pos_sum = XSUM(pos);
    // pos_sum = sum(pos * pos)
    
    float inside = (p*p / 4 + rsq) - pos_sum;
    if (inside < 0) { res->success = 0; continue; }
    
    float sq = sqrtf(inside),
      k = - p / 2,
      k1 = k + sq,
      k2 = k - sq;
    */
    if (k1 < 0) { res->success = 0; continue; }
    
    res->success = 1;
    
    if (k2 > 0) res->distance = k2;
    else res->distance = k1;
    
    // col = (1, 1, 1)
    res->emissive_col = (vec3f){0,0,0,0};
    res->col = (vec3f){1,1,1,1};
    
    v4sf normal = V4SF(RAY->pos) + V4SF(RAY->dir) * (v4sf) FOUR(res->distance);
    // normal = (ray.pos + distance * ray.dir) - center
    normal = normal - V4SF(center);
    // normalize normal
    v4sf nprod = normal * normal;
    float nprodf = 1.0f / sqrtf(*(float*) &nprod + *((float*) &nprod + 1) + *((float*) &nprod + 2));
    // nprod = __builtin_ia32_rsqrtss(SUM(nprod));
    // normal *= __builtin_ia32_shufps(nprod, nprod, 0x0);
    normal *= (v4sf) FOUR(nprodf);
    V4SF(res->normal) = normal;
  }
}

void fast_scale_process(
  struct Ray **rayplanes, struct Result **resplanes,
  struct VMState *states, int numstates,
  float factor,
  void* self
) {
  for (int i = 0; i < numstates; ++i) {
    struct VMState* sp = states++;
    
    if (sp->stream_ptr[0] != self) continue;
    sp->stream_ptr ++; sp->stream_len --;
    
    struct Ray *RAY  = rayplanes[sp->rayid - 1] + i;
    sp->rayid ++;
    struct Ray *RAY2 = rayplanes[sp->rayid - 1] + i;
    
    V4SF(RAY2->pos) = V4SF(RAY->pos) * (v4sf) FOUR(1/factor);
    V4SF(RAY2->dir) = V4SF(RAY->dir);
  }
}

#include <limits.h>
void ALIGNED fast_checker_process(
  struct Ray **rayplanes, struct Result **resplanes,
  struct VMState *states, int numstates,
  vec3f a, vec3f b,
  void* self
) {
  for (int i = 0; i < numstates; ++i) {
    struct VMState* sp = states++;
    
    if (sp->stream_ptr[0] != self) continue;
    sp->stream_ptr ++; sp->stream_len --;
    
    struct Result *res = resplanes[sp->resid - 1] + i;
    struct Ray *ray = rayplanes[sp->rayid - 1] + i;
    if (res -> success) {
      v4sf hitpos = V4SF(ray->pos) + (v4sf) FOUR(res->distance) * V4SF(ray->dir);
      vec3f hitposv = *(vec3f*) &hitpos;
      res->emissive_col = (vec3f){0,0,0,0};
      // would overflow
      if (fabsf(hitposv.x) > INT_MAX || fabsf(hitposv.y) > INT_MAX || fabsf(hitposv.z) > INT_MAX) {
        v4sf temp = (V4SF(b) + V4SF(a)) / (v4sf) FOUR(2);
        res->col = *(vec3f*) &temp;
        continue;
      }
      int ix = (int) hitposv.x, iy = (int) hitposv.y, iz = (int) hitposv.z;
      if (hitposv.x < 0) ix --;
      if (hitposv.y < 0) iy --;
      if (hitposv.z < 0) iz --;
      if ((ix & 1) ^ (iy & 1) ^ (iz & 1))
        res->col = b;
      else
        res->col = a;
    }
  }
}

void ALIGNED fast_plane_process(
  struct Ray **rayplanes, struct Result **resplanes,
  struct VMState *states, int numstates,
  vec3f normal, vec3f base,
  void* self
) {
  for (int i = 0; i < numstates; ++i) {
    struct VMState* sp = states++;
    
    if (sp->stream_ptr[0] != self) continue;
    sp->stream_ptr ++; sp->stream_len --;
    
    struct Ray *ray = rayplanes[sp->rayid - 1] + i;
    v4sf pos = V4SF(ray->pos), dir = V4SF(ray->dir);
    
    struct Result *res = resplanes[++sp->resid - 1] + i;
    v4sf part1 = V4SF(normal) * (pos - V4SF(base));
    v4sf part2 = dir * V4SF(normal);
    float sum2 = XSUM(part2);
    if (sum2 >= 0) { res->success = 0; continue; } // hit plane from behind - ignore
    float dist = -XSUM(part1) / sum2;
    if (dist < 0) res->success = 0;
    else {
      res->success = 1;
      res->distance = dist;
      res->col = (vec3f){1,1,1,1};
      res->emissive_col = (vec3f){0,0,0,0};
      res->normal = normal;
    }
  }
}

void ALIGNED fast_group_process(
  struct Ray **rayplanes, struct Result **resplanes,
  struct VMState *states, int numstates,
  int len,
  void* self
) {
  if (UNLIKELY(!len)) {
    for (int i = 0; i < numstates; ++i) {
      struct VMState* sp = states+i;
      
      if (sp->stream_ptr[0] != self) continue;
      sp->stream_ptr ++; sp->stream_len --;
      
      sp->resid ++;
      resplanes[sp->resid-1][i].success = 0;
    }
    return;
  }
  for (int i = 0; i < numstates; ++i) {
    struct VMState* sp = states+i;
    
    if (sp->stream_ptr[0] != self) continue;
    sp->stream_ptr ++; sp->stream_len --;
    
    PREFETCH_HARD(&resplanes[sp->resid-2][i].success, 0, 0);
    PREFETCH_HARD(&resplanes[sp->resid-1][i+1].success, 0, 0);
    if (!resplanes[sp->resid-1][i].success) { }
    else if (!resplanes[sp->resid-2][i].success) {
      resplanes[sp->resid-2][i] = resplanes[sp->resid-1][i];
    } else {
      if (resplanes[sp->resid-2][i].distance <= resplanes[sp->resid-1][i].distance) {
      } else {
        resplanes[sp->resid-2][i] = resplanes[sp->resid-1][i];
      }
    }
    sp->resid --;
    continue;
  }
}

void ALIGNED fast_translate_process(
  struct Ray **rayplanes, struct Result **resplanes,
  struct VMState *states, int numstates,
  vec3f vector,
  void* self
) {
  for (int i = 0; i < numstates; ++i) {
    struct VMState* sp = states++;
    
    if (sp->stream_ptr[0] != self) continue;
    sp->stream_ptr ++; sp->stream_len --;
    
    struct Ray *ray  = rayplanes[sp->rayid - 1] + i;
    sp->rayid ++;
    struct Ray *ray2 = rayplanes[sp->rayid - 1] + i;
    V4SF(ray2->pos) = V4SF(ray->pos) - V4SF(vector);
    V4SF(ray2->dir) = V4SF(ray->dir);
  }
}

void ALIGNED fast_light_process(
  struct Ray **rayplanes, struct Result **resplanes,
  struct VMState *states, int numstates,
  vec3f* lightpos,
  void* self
) {
  for (int i = 0; i < numstates; ++i) {
    struct VMState* sp = states++;
    
    if (sp->stream_ptr[0] != self) continue;
    sp->stream_ptr ++; sp->stream_len --;
    
    struct Result *res = resplanes[sp->resid - 1] + i;
    if (res->success) {
      v4sf nspos;
      {
        struct Ray *ray = rayplanes[sp->rayid - 1] + i;
        nspos = V4SF(ray->pos) + V4SF(ray->dir) * (v4sf) FOUR(res->distance * 0.999);
      }
      sp->rayid ++;
      v4sf lightdir = *(v4sf*) lightpos - nspos;
      v4sf lsq = lightdir * lightdir;
      float ldfac = 1 / sqrtf(XSUM(lsq));
      lightdir *= (v4sf) FOUR(ldfac);
      {
        struct Ray *ray = rayplanes[sp->rayid - 1] + i;
        V4SF(ray->pos) = nspos;
        V4SF(ray->dir) = lightdir;
      }
    } else {
      sp->stream_ptr += sp->stream_len - 1;
      sp->stream_len = 1;
    }
  }
}

typedef struct {
  vec3f a, b, c, n;
  vec2f uv_a, uv_ba, uv_ca;
  float invDenom; void *texstate;
} TriangleInfo;

typedef struct {
  vec3f a, b;
} AABB;

typedef struct TriangleNode {
  AABB aabb;
  int children_length; struct TriangleNode **children_ptr;
  int capacity, length; int *info;
} TriangleNode;

static int internal_rayHitsAABB(vec3f *abp, vec3f *p_ray, float *dist) {
#define ap &abp[0]
#define bp &abp[1]
#define p_pos &p_ray[0]
#define p_dir &p_ray[1]
  #define SF(VAR) (*(v4sf*) VAR)
  float dirprod = X(SF(p_dir)) * Y(SF(p_dir)) * Z(SF(p_dir));
  #undef SF
  v4si mask = (v4si) FOUR(1<<31);
  v4si signs = mask & *(v4si*)p_dir;
  v4sf a = (v4sf) (V4SI(*ap) ^ signs);
  v4sf b = (v4sf) (V4SI(*bp) ^ signs);
  v4sf pos = (v4sf) (V4SI(*p_pos) ^ signs);
  v4sf dir = (v4sf) (V4SI(*p_dir) ^ signs);
  v4sf b_ = b;
  // pretend ray starts at origin: -pos
  b = __builtin_ia32_maxps(a, b) - pos;
  
  // if (X(b) < 0 || Y(b) < 0 || Z(b) < 0) return 0; // ray is pointed away from aabb.
  v4si bsign = mask & *(v4si*)&b;
  if (IX(bsign) | IY(bsign) | IZ(bsign)) return 0;
  
  a = __builtin_ia32_minps(a, b_) - pos;
  // multiply every component with dir.(x*y*z)
  // vec3f dista = a / dir, distb = b / dir;
  vec3f *_vdir = (vec3f*) &dir;
  vec3f *_dista = (vec3f*) &a, *_distb = (vec3f*) &b;
#define vdir (*_vdir)
#define dista (*_dista)
#define distb (*_distb)
  
  if (LIKELY(vdir.x != 0 && vdir.y != 0 && vdir.z != 0)) {
    // vdir += (v4sf) {0, 0, 0, 1};
    *(v4si*) &dir &= (v4si) {-1, -1, -1, 0};
    dir += (v4sf) {0, 0, 0, 1};
    a /= dir;
    b /= dir;
  } else {
    if (LIKELY(vdir.x != 0)) { dista.x /= vdir.x; distb.x /= vdir.x; }
    else { dista.x = copysignf(INFINITY, dista.x); distb.x = copysignf(INFINITY, distb.x); }
    
    if (LIKELY(vdir.y != 0)) { dista.y /= vdir.y; distb.y /= vdir.y; }
    else { dista.y = copysignf(INFINITY, dista.y); distb.y = copysignf(INFINITY, distb.y); }
    
    if (LIKELY(vdir.z != 0)) { dista.z /= vdir.z; distb.z /= vdir.z; }
    else { dista.z = copysignf(INFINITY, dista.z); distb.z = copysignf(INFINITY, distb.z); }
  }
  float entry = fmaxf(dista.x, fmaxf(dista.y, dista.z));
  float exit = fminf(distb.x, fminf(distb.y, distb.z));
  if (dist) { *dist = entry; }
  return entry <= exit;
#undef dista
#undef vdir
#undef ap
#undef bp
#undef p_pos
#undef p_dir
}

int ALIGNED fast_rayHitsAABB(vec3f *abp, vec3f *p_ray, float *dist) {
  return internal_rayHitsAABB(abp, p_ray, dist);
}

static int rayHits(AABB *aabb, vec3f *ray, float *dist) {
  return internal_rayHitsAABB(&aabb->a, ray, dist);
}

typedef struct {
  vec3f pos, dir;
  TriangleInfo* closest_res;
  float closest;
  vec2f uv;
} RecursionInfo;

#include <float.h>

#define ROUND16(X) ((void*) ((unsigned int)((char*) (X) + 15) & ~15))

static void __attribute__ ((hot)) internal_triangle_recurse(TriangleNode *node, TriangleInfo *tlist, RecursionInfo *info, int *cache, int hash) {
  // __builtin_prefetch(&((TriangleInfo*) ROUND16(node + 1))->a);
  // printf("early prefetch %p\n", &((TriangleInfo*) ROUND16(node+1))->a);
  if (node->children_length) {
    if (info->closest_res) {
      float fs;
      for (int i = 0; i < node->children_length; ++i) {
        if (rayHits(&node->children_ptr[i]->aabb, &info->pos, &fs) && fs < info->closest)
          internal_triangle_recurse(node->children_ptr[i], tlist, info, cache, hash);
      }
    } else {
      for (int i = 0; i < node->children_length; ++i) {
        if (rayHits(&node->children_ptr[i]->aabb, &info->pos, 0)) internal_triangle_recurse(node->children_ptr[i], tlist, info, cache, hash);
      }
    }
  } else {
    RecursionInfo rin = *info;
    for (int i = 0; i < node->length; ++i) {
      int id = node->info[i];
      // printf("%i: hash %i, compare %i: outcome %i\n", id, hash, cache[id].hash, cache[id].outcome);
      if (cache[id] == hash) continue; // already considered
      cache[id] = hash;
      TriangleInfo *ti = &tlist[id];
      v4sf v_1 = V4SF(ti->n) * (V4SF(rin.pos) - V4SF(ti->a));
      v4sf v_2 = V4SF(rin.dir) * V4SF(ti->n);
      // float dist = - XSUM(v_1) / XSUM(v_2);
      // if (dist < 0) continue;
      float f_1 = XSUM(v_1), f_2 = XSUM(v_2);
      if (f_2 == 0) continue;
      
      // float dist = - f_1 / f_2;
      // if (dist < 0 || dist > rin.closest) continue;
      if (- f_1 * f_2 < 0) continue;
      float dist = - f_1 / f_2;
      
      if (UNLIKELY(dist > rin.closest)) continue;
      
      v4sf p = V4SF(rin.pos) + (v4sf) FOUR(dist) * V4SF(rin.dir);
      v4sf v0 = V4SF(ti->c) - V4SF(ti->a);
      v4sf v1 = V4SF(ti->b) - V4SF(ti->a);
      v4sf v2 = p           - V4SF(ti->a);
      v4sf v00 = v0 * v0, v01 = v0 * v1, v11 = v1 * v1, v02 = v0 * v2, v12 = v1 * v2;
      float dot00 = XSUM(v00), dot01 = XSUM(v01), dot11 = XSUM(v11);
      float dot02 = XSUM(v02), dot12 = XSUM(v12);
      float invDenom = ti->invDenom;
      
      v4sf bogus;
      v4sf temp = __builtin_ia32_hsubps((v4sf) { dot11, dot01, dot00, dot01 } * (v4sf) { dot02, dot12, dot12, dot02 }, bogus);
      float u = X(temp) * invDenom;
      float v = Y(temp) * invDenom;
      // float u = X(__builtin_ia32_hsubps((v4sf) { dot11, dot01 } * (v4sf) { dot02, dot12 }, bogus)) * invDenom;
      // float v = X(__builtin_ia32_hsubps((v4sf) { dot00, dot01 } * (v4sf) { dot12, dot02 }, bogus)) * invDenom;
      // float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
      // float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
      if (UNLIKELY((u > 0) && (v > 0) && (u+v < 1))) {
        rin.closest_res = ti;
        rin.closest = dist;
        rin.uv = (vec2f) {u, v};
      }
    }
    *info = rin;
  }
}

static void fast_triangle_recurse_intern(TriangleNode *node, vec3f *pos, vec3f *dir, TriangleInfo *tlist, TriangleInfo **closest_res, float *closest, vec2f *uv,
                                         int *cache, int hash) {
  __attribute__ ((aligned (16))) RecursionInfo ri;
  *(v4sf*) &ri.pos = *(v4sf*) pos;
  *(v4sf*) &ri.dir = *(v4sf*) dir;
  ri.closest_res = 0;
  ri.closest = INFINITY;
  internal_triangle_recurse(node, tlist, &ri, cache, hash);
  *closest_res = ri.closest_res;
  *closest = ri.closest;
  *uv = ri.uv;
}

struct HdrTex {
  int w, h;
  int data_len; v4sf* data_ptr;
};

struct Texture {
  gdImagePtr gdp;
  struct HdrTex *hdp;
};

static v4sf lookupTex(int x, int y, struct Texture* texptr) {
  if (texptr->gdp) {
    int res = texptr->gdp->tpixels[((unsigned int) y)%texptr->gdp->sy][((unsigned int) x)%texptr->gdp->sx];
    return (v4sf){(res>>16)&0xff, (res>>8)&0xff, (res>>0)&0xff, 0} / (v4sf){256,256,256,256};
  } else {
    int index = y * texptr->hdp->w + x;
    if (index < 0 || index >= texptr->hdp->data_len) return (v4sf) FOUR(0);
    return texptr->hdp->data_ptr[index];
  }
}

void ALIGNED interpolate(float u, float v, struct Texture* texptr, vec3f *res) {
  int w, h;
  if (texptr->gdp) {
    w = texptr->gdp->sx;
    h = texptr->gdp->sy;
  } else {
    w = texptr->hdp->w;
    h = texptr->hdp->h;
  }
  float coordx = u * w, coordy = v * h;
  int ix = (int) floorf(coordx), iy = (int) floorf(coordy);
  float facx = coordx - ix, facy = coordy - iy, ifacx = 1 - facx, ifacy = 1 - facy;
#define MKV4SF(X) ({ float f = (X); (v4sf) {f,f,f,f}; })
#define V4RES (*(v4sf*) res)
  V4RES =
      lookupTex(ix, iy, texptr)     * MKV4SF(ifacx * ifacy)
    + lookupTex(ix, iy+1, texptr)   * MKV4SF(ifacx *  facy)
    + lookupTex(ix+1, iy, texptr)   * MKV4SF( facx * ifacy)
    + lookupTex(ix+1, iy+1, texptr) * MKV4SF( facx *  facy)
  ;
  // blatantly cheat the envmap detection
  V4RES = (v4sf)FOUR(1.0f/255) + V4RES * (v4sf)FOUR(254.0f/255);
#undef MKV4SF
}

void ALIGNED fast_triangleset_process(
  struct Ray **rayplanes, struct Result **resplanes,
  struct VMState *states, int numstates,
  TriangleInfo *tlist, TriangleNode *root,
  int *cache, int *hashp,
  void* self
) {
  for (int i = 0; i < numstates; ++i) {
    struct VMState* sp = states + i;
    
    if (sp->stream_ptr[0] != self) continue;
    sp->stream_ptr ++; sp->stream_len --;
    
    sp->resid ++;
    struct Result *res = resplanes[sp->resid-1] + i;
    
    vec3f *pos = &rayplanes[sp->rayid-1][i].pos;
    vec3f *dir = &rayplanes[sp->rayid-1][i].dir;
    
    PREFETCH_HARD(pos, 0, 3);
    PREFETCH_HARD(res, 1, 3);
    
    float closest; TriangleInfo *closest_info = 0;
    vec2f texcoord;
    (*hashp) ++;
    fast_triangle_recurse_intern(root, pos, dir, tlist, &closest_info, &closest, &texcoord, cache, *hashp);
    
    res->success = 0;
    if (closest_info) {
      vec2f texcoord2;
      texcoord2.x = closest_info->uv_a.x + texcoord.x * closest_info->uv_ca.x + texcoord.y * closest_info->uv_ba.x;
      texcoord2.y = closest_info->uv_a.y + texcoord.x * closest_info->uv_ca.y + texcoord.y * closest_info->uv_ba.y;
      res->texcoord = texcoord2;
      res->success = 1;
      res->distance = closest;
      res->emissive_col = (vec3f) {0,0,0,0};
      res->normal = closest_info->n;
      void* texst = closest_info->texstate;
      res->texinfo = texst;
      if (texst) {
        interpolate(texcoord2.x, 1-texcoord2.y, texst, &res->col);
      } else {
        res->col = (vec3f) {1,1,1,1};
      }
    }
  }
}
