#include "stdio.h"
#include "math.h"

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
  int success;
};

struct VMState {
  int resid, rayid;
  struct Ray *rays_ptr;
  struct Result *res_ptr;
  int stream_len; void** stream_ptr;
  int x, y;
};

typedef float v4sf __attribute__ ((vector_size (16)));
typedef int v4si __attribute__ ((vector_size (16)));

#define ALIGNED __attribute__ ((force_align_arg_pointer))

void ALIGNED fastsetup(int xfrom, int xto, int yfrom, int yto, int dw, int dh, struct VMState* state) {
  float ratio = dw * 1.0f / dh;
  for (int y = yfrom; y < yto; ++y) {
    for (int x = xfrom; x < xto; ++x) {
      state->resid = 0;
      state->rayid = 1;
      float fx = x, fy = y;
      // use dedicated step for this
      // if (jitter) { fx += neat_randf(randfparam) - 0.5; fy += neat_randf(randfparam) - 0.5; }
      v4sf v = (v4sf) {ratio * (fx / (dw / 2.0) - 1.0), 1.0 - fy / (dh / 2.0), 1.0, 0.0};
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
#define Y(vec) __builtin_ia32_vec_ext_v4sf ((vec), 1)
#define Z(vec) __builtin_ia32_vec_ext_v4sf ((vec), 2)
#define W(vec) __builtin_ia32_vec_ext_v4sf ((vec), 3)
#define IX(vec) __builtin_ia32_vec_ext_v4si ((vec), 0)
#define IY(vec) __builtin_ia32_vec_ext_v4si ((vec), 1)
#define IZ(vec) __builtin_ia32_vec_ext_v4si ((vec), 2)
#define IW(vec) __builtin_ia32_vec_ext_v4si ((vec), 3)
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

// IMPORTANT: use -mstackrealign!
void ALIGNED fast_sphere_process(
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

#include <limits.h>
void ALIGNED fast_checker_process(
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
      res->emissive_col = (vec3f){0,0,0,0};
      if (fabsf(hitposv.x) > INT_MAX || fabsf(hitposv.y) > INT_MAX || fabsf(hitposv.z) > INT_MAX) {
        res->col = a;
      } else {
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
}


void ALIGNED fast_plane_process(
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

void ALIGNED fast_translate_process(
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

void ALIGNED fast_light_process(
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
  int capacity, length; TriangleInfo *info;
} TriangleNode;

#define LIKELY(X) (__builtin_expect((X), 1))
#define UNLIKELY(X) (__builtin_expect((X), 0))

static int internal_rayHitsAABB1d(float a, float b, float pos, float dir, float *dist) {
  if (dir < 0) { a = -a; b = -b; pos = -pos; dir = -dir; }
  float a_ = fminf(a, b) - pos;
  b = fmaxf(a, b) - pos;
  a = a_;
  if (b < 0) return 0;
  
  if (dist) *dist = a / dir;
  return 1;
}

#define UINT(F) ({ float f = F; *(unsigned int*) &f; })
#define FLOAT(I) ({ unsigned int i = I; *(float*) &i; })

static int internal_rayHitsAABB2d(v4sf ab, v4sf ray, float *dist) {
  if (UNLIKELY(Z(ray) == 0)) {
    if (X(ray) < X(ab) || X(ray) > Z(ab)) return 0;
    return internal_rayHitsAABB1d(Y(ab), W(ab), Y(ray), W(ray), dist);
  }
  if (UNLIKELY(W(ray) == 0)) {
    if (Y(ray) < Y(ab) || Y(ray) > W(ab)) return 0;
    return internal_rayHitsAABB1d(X(ab), Z(ab), X(ray), Z(ray), dist);
  }
  v4si mask = (v4si) FOUR(1<<31);
  v4sf temp = (v4sf){Z(ray), W(ray), Z(ray), W(ray)};
  v4si signs = mask & *(v4si*) &temp;
  ab = (v4sf) (*(v4si*) &ab ^ signs);
  ray = (v4sf) (*(v4si*) &ray ^ signs);
  ab = (v4sf) {fminf(X(ab), Z(ab)), fminf(Y(ab), W(ab)), fmaxf(X(ab), Z(ab)), fmaxf(Y(ab), W(ab))} - (v4sf) {X(ray), Y(ray), X(ray), Y(ray)};
  v4si absign = mask & *(v4si*)&ab;
  if (IZ(absign) | IW(absign)) return 0;
  
  // multiply every component with dir.(x*y)
  // vec3f distab = ab / {dir, dir};
  v4sf mulfac = __builtin_ia32_shufps(ray, ray, 238) /* wzwz */;
  v4sf distab = ab * mulfac;
  
  float entry = fmaxf(X(distab), Y(distab));
  float exit = fminf(Z(distab), W(distab));
  if (dist) { *dist = entry / (Z(ray) * W(ray)); }
  return entry <= exit;
}

static int internal_rayHitsAABB(vec3f *abp, vec3f *p_ray, float *dist) {
#define ap &abp[0]
#define bp &abp[1]
#define p_pos &p_ray[0]
#define p_dir &p_ray[1]
  #define SF(VAR) (*(v4sf*) VAR)
  float dirprod = X(SF(p_dir)) * Y(SF(p_dir)) * Z(SF(p_dir));
  if (UNLIKELY(dirprod == 0)) {
    if (UNLIKELY(X(SF(p_dir)) == 0)) {
      if (X(SF(p_pos)) < X(SF(ap)) || X(SF(p_pos)) > X(SF(bp))) return 0;
      v4sf ab = (v4sf) {Y(SF(ap)), Z(SF(ap)), Y(SF(bp)), Z(SF(bp))};
      v4sf ray = (v4sf) {Y(SF(p_pos)), Z(SF(p_pos)), Y(SF(p_dir)), Z(SF(p_dir))};
      return internal_rayHitsAABB2d(ab, ray, dist);
    }
    if (UNLIKELY(Y(SF(p_dir)) == 0)) {
      if (Y(SF(p_pos)) < Y(SF(ap)) || Y(SF(p_pos)) > Y(SF(bp))) return 0;
      v4sf ab = (v4sf) {X(SF(ap)), Z(SF(ap)), X(SF(bp)), Z(SF(bp))};
      v4sf ray = (v4sf) {X(SF(p_pos)), Z(SF(p_pos)), X(SF(p_dir)), Z(SF(p_dir))};
      return internal_rayHitsAABB2d(ab, ray, dist);
    }
    if (UNLIKELY(Z(SF(p_dir)) == 0)) {
      if (Z(SF(p_pos)) < Z(SF(ap)) || Z(SF(p_pos)) > Z(SF(bp))) return 0;
      v4sf ab = (v4sf) {X(SF(ap)), Y(SF(ap)), X(SF(bp)), Y(SF(bp))};
      v4sf ray = (v4sf) {X(SF(p_pos)), Y(SF(p_pos)), X(SF(p_dir)), Y(SF(p_dir))};
      return internal_rayHitsAABB2d(ab, ray, dist);
    }
    __asm__("int $3");
    return 0; // should never happen, only here for gcc flow control
  }
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
  v4sf mulfac = __builtin_ia32_shufps(dir, dir, 1) /* yxx */ * __builtin_ia32_shufps(dir, dir, 26) /* zzy */;
  v4sf dista = a * mulfac, distb = b * mulfac;
  
  float entry = fmaxf(X(dista), fmaxf(Y(dista), Z(dista)));
  float exit = fminf(X(distb), fminf(Y(distb), Z(distb)));
  if (dist) { *dist = entry / fabsf(dirprod); }
  return entry <= exit;
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

#define EPS 0.1

static void __attribute__ ((hot)) internal_triangle_recurse(TriangleNode *node, RecursionInfo *info) {
  // __builtin_prefetch(&((TriangleInfo*) ROUND16(node + 1))->a);
  // printf("early prefetch %p\n", &((TriangleInfo*) ROUND16(node+1))->a);
  if (node->children_length) {
    if (info->closest_res) {
      float fs;
      for (int i = 0; i < node->children_length; ++i) {
        if (rayHits(&node->children_ptr[i]->aabb, &info->pos, &fs) && fs < info->closest)
          internal_triangle_recurse(node->children_ptr[i], info);
      }
    } else {
      for (int i = 0; i < node->children_length; ++i) {
        if (rayHits(&node->children_ptr[i]->aabb, &info->pos, 0)) internal_triangle_recurse(node->children_ptr[i], info);
      }
    }
  } else {
    // printf("late prefetch %p\n", (void*) &node->info[0]);
    // already prefetched up top
    // __builtin_prefetch(&node->info[0]);
    RecursionInfo rin = *info;
    for (int i = 0; i < node->length; ++i) {
      // __builtin_prefetch(&node->info[i+1]);
      TriangleInfo *ti = &node->info[i];
      // printf("now access %p\n", (void*) &ti->a);
      // __asm__("int $3");
      v4sf v_1 = V4SF(ti->n) * (V4SF(rin.pos) - V4SF(ti->a));
      v4sf v_2 = V4SF(rin.dir) * V4SF(ti->n);
      // float dist = - XSUM(v_1) / XSUM(v_2);
      // if (dist < 0) continue;
      float f_1 = XSUM(v_1), f_2 = XSUM(v_2);
      if (f_2 == 0) continue;
      float dist = - f_1 / f_2;
      // if (
      //   f_2 >= 0 /* otherwise, backside hit */ ||
      //   f_1 < 0) continue;
      // float dist = - f_1 / f_2;
      // don't need to retest this since rin->closest starts out FLT_MAX
      // if (LIKELY(-rin.closest * f_2 < f_1)) continue;
      if (dist < 0 || dist > rin.closest) continue;
      // if (dist < 0.1) continue;
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
      int test = (u > 0) & (v > 0) & (u+v < 1);
      if (UNLIKELY(test)) {
        rin.closest_res = ti;
        rin.closest = dist;
        rin.uv = (vec2f) {u, v};
      }
    }
    *info = rin;
  }
}

void ALIGNED fast_triangle_recurse(TriangleNode *node, vec3f *pos, vec3f *dir, TriangleInfo** closest_res, float *closest, vec2f *uv) {
  RecursionInfo ri;
  ri.pos = *pos; ri.dir = *dir;
  ri.closest_res = 0;
  ri.closest = INFINITY;
  internal_triangle_recurse(node, &ri);
  *closest_res = ri.closest_res;
  *closest = ri.closest;
  *uv = ri.uv;
}
