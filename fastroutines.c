#include "stdio.h"
#include "math.h"

typedef struct {
  float x, y, z, w;
} __attribute__ ((aligned (16))) vec3f;

struct Ray {
  vec3f pos, dir;
};

struct Result {
  float distance;
  vec3f normal, col;
  int success;
};

struct VMState {
  int resid, rayid;
  int rays_length; struct Ray *rays_ptr;
  int res_length; struct Result *res_ptr;
  int stream_len; void** stream_ptr;
};

typedef float v4sf __attribute__ ((vector_size (16)));

void fastsetup(int yfrom, int yto, int dw, int dh, struct VMState* state) {
  for (int y = yfrom; y < yto; ++y) {
    for (int x = 0; x < dw; ++x) {
      state->resid = 0;
      state->rayid = 1;
      v4sf v = (v4sf) {x / (dw / 2.0) - 1.0, 1.0 - y / (dh / 2.0), 1.0, 0.0};
      v4sf res = v;
      // v = v * v
      v *= v;
      // res /= sqrt(v + v.yy + v.zz)
      float f = 1.0f / sqrtf(*(float*) &v + *((float*) &v + 1) + *((float*) &v + 2));
      /*v = __builtin_ia32_rsqrtss(
        v + __builtin_ia32_shufps(v, v, 0x55)
          + __builtin_ia32_shufps(v, v, 0xaa)
      );
      res *= __builtin_ia32_shufps(v, v, 0x0);*/
      res *= (v4sf) {f, f, f, f};
      *(v4sf*)&state->rays_ptr->dir = res;
      *(v4sf*)&state->rays_ptr->pos = (v4sf){0,2,0,0};
      state ++;
    }
  }
}

#define FOUR(x){x,x,x,x}

#define SUM(vec) \
  ((vec)\
     + __builtin_ia32_shufps((vec), (vec), 0x55)\
     + __builtin_ia32_shufps((vec), (vec), 0xaa)\
  )
#define X(vec) __builtin_ia32_vec_ext_v4sf ((vec), 0)
// #define XSUM(vec) X(SUM(vec))
/*#define XSUM(vec) (\
    __builtin_ia32_vec_ext_v4sf ((vec), 0) \
  + __builtin_ia32_vec_ext_v4sf ((vec), 1) \
  + __builtin_ia32_vec_ext_v4sf ((vec), 2))*/
/*#define XSUM(vec) __builtin_ia32_vec_ext_v4sf(\
  __builtin_ia32_shufps((vec), (vec), 0xaa)\
+ __builtin_ia32_haddps((vec), (vec))\
, 0)*/
#define XSUM(vec) (*(float*) &(vec) + *((float*) &(vec) + 1) + *((float*) &(vec) + 2))
#define XYZ(v) (v).x, (v).y, (v).z
#define V4SF(v) (*(v4sf*) &(v))

// IMPORTANT: use -mstackrealign!
void fast_sphere_process(
  struct VMState *states, int numstates,
  vec3f center, float rsq,
  void* self
) {
#define PREFETCH_HARD(X, READ, LOCALITY) \
  __builtin_prefetch(X, READ, LOCALITY); \
  __asm__ volatile ("" : : : "memory"); // force break
  for (int i = 0; i < numstates; ++i) {
    struct VMState* sp = states++;
    PREFETCH_HARD(sp, 1, 3);
    
    if (sp->stream_ptr[0] != self) continue;
    struct Ray* RAY = sp->rays_ptr + sp->rayid - 1;
    
    sp->stream_ptr ++; sp->stream_len --;
    struct Result *res = sp->res_ptr + sp->resid ++;
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
  struct VMState *states, int numstates,
  float factor,
  void* self
) {
  for (int i = 0; i < numstates; ++i) {
    struct VMState* sp = states++;
    
    if (sp->stream_ptr[0] != self) continue;
    sp->stream_ptr ++; sp->stream_len --;
    
    struct Ray *RAY  = sp->rays_ptr + sp->rayid - 1;
    sp->rayid ++;
    struct Ray *RAY2 = sp->rays_ptr + sp->rayid - 1;
    
    V4SF(RAY2->pos) = V4SF(RAY->pos) * (v4sf) FOUR(1/factor);
    V4SF(RAY2->dir) = V4SF(RAY->dir);
  }
}

void fast_checker_process(
  struct VMState *states, int numstates,
  vec3f a, vec3f b,
  void* self
) {
  for (int i = 0; i < numstates; ++i) {
    struct VMState* sp = states++;
    
    if (sp->stream_ptr[0] != self) continue;
    sp->stream_ptr ++; sp->stream_len --;
    
    struct Result *res = sp->res_ptr + sp->resid - 1;
    struct Ray *ray = sp->rays_ptr + sp->rayid - 1;
    if (res -> success) {
      v4sf hitpos = V4SF(ray->pos) + (v4sf) FOUR(res->distance) * V4SF(ray->dir);
      vec3f hitposv = *(vec3f*) &hitpos;
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


void fast_plane_process(
  struct VMState *states, int numstates,
  vec3f normal, vec3f base,
  void* self
) {
  for (int i = 0; i < numstates; ++i) {
    struct VMState* sp = states++;
    
    if (sp->stream_ptr[0] != self) continue;
    sp->stream_ptr ++; sp->stream_len --;
    
    struct Ray *ray = sp->rays_ptr + sp->rayid - 1;
    v4sf pos = V4SF(ray->pos), dir = V4SF(ray->dir);
    
    struct Result *res = sp->res_ptr + ++ sp->resid - 1;
    v4sf part1 = V4SF(normal) * (pos - V4SF(base));
    v4sf part2 = dir * V4SF(normal);
    float dist = -XSUM(part1) / XSUM(part2);
    if (dist < 0) res->success = 0;
    else {
      res->success = 1;
      res->distance = dist;
      V4SF(res->col) = (v4sf) FOUR(1);
      res->normal = normal;
    }
  }
}

void fast_group_process(
  struct VMState *states, int numstates,
  int len,
  void* self
) {
  for (int i = 0; i < numstates; ++i) {
    struct VMState* sp = states++;
    
    if (sp->stream_ptr[0] != self) continue;
    sp->stream_ptr ++; sp->stream_len --;
    
    int match = -1;
    int resid = sp->resid, delta = resid - len;
    struct Result *res = sp->res_ptr + delta;
    float lowestDistance;
    for (int k = 0; k < len; ++k) {
      struct Result *current = res + k;
      if (current->success && (match == -1 || current->distance < lowestDistance)) {
        lowestDistance = current->distance;
        match = k;
      }
    }
    if (match != -1) {
      sp->res_ptr[delta] = res[match];
    } else {
      sp->res_ptr[delta].success = 0;
    }
    sp->resid -= len - 1;
  }
}

void fast_translate_process(
  struct VMState *states, int numstates,
  vec3f vector,
  void* self
) {
  for (int i = 0; i < numstates; ++i) {
    struct VMState* sp = states++;
    
    if (sp->stream_ptr[0] != self) continue;
    sp->stream_ptr ++; sp->stream_len --;
    
    struct Ray *ray = sp->rays_ptr + sp->rayid - 1;
    sp->rayid ++;
    struct Ray *ray2 = sp->rays_ptr + sp->rayid - 1;
    V4SF(ray2->pos) = V4SF(ray->pos) - V4SF(vector);
    V4SF(ray2->dir) = V4SF(ray->dir);
  }
}

void fast_light_process(
  struct VMState *states, int numstates,
  vec3f* lightpos,
  void* self
) {
  for (int i = 0; i < numstates; ++i) {
    struct VMState* sp = states++;
    
    if (sp->stream_ptr[0] != self) continue;
    sp->stream_ptr ++; sp->stream_len --;
    
    struct Result *res = sp->res_ptr + sp->resid - 1;
    if (res->success) {
      v4sf nspos;
      {
        struct Ray *ray = sp->rays_ptr + sp->rayid - 1;
        nspos = V4SF(ray->pos) + V4SF(ray->dir) * (v4sf) FOUR(res->distance * 0.999);
      }
      sp->rayid ++;
      v4sf lightdir = *(v4sf*) lightpos - nspos;
      v4sf lsq = lightdir * lightdir;
      float ldfac = 1 / sqrtf(XSUM(lsq));
      lightdir *= (v4sf) FOUR(ldfac);
      {
        struct Ray *ray = sp->rays_ptr + sp->rayid - 1;
        V4SF(ray->pos) = nspos;
        V4SF(ray->dir) = lightdir;
      }
    } else {
      sp->stream_ptr += sp->stream_len - 1;
      sp->stream_len = 1;
    }
  }
}
