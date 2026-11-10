// pegeq: builds the extensive-form game for an EQUILIBRIUM pre-endgame solve.
//
// Model. All tiles not on the board form the pool U. Chance deals the mover's
// rack, the opponent's rack and the bag uniformly from U (every tile-level
// assignment equally likely), so neither player knows the other's rack. Play
// proceeds turn by turn; a play draws replacement tiles from the bag (a chance
// node only the drawing player observes). As soon as the bag is empty every
// tile is accounted for, both racks are common knowledge, and the rest of the
// game is a perfect-information endgame scored exactly by MAGPIE's endgame
// solver. Those endgame results are the leaves.
//
// Deal set. The full uniform prior is usually too large, so deals are taken
// from the level-L closure around the mover's real rack R0 (as in sscg13's
// "level 2 information set"): level 1 pairs R0 with every opponent rack the
// mover cannot rule out; level 2 adds, for each of those opponent racks, every
// mover rack that opponent cannot rule out; and so on. -level 0 uses every
// deal (the exact uniform-prior game).
//
// Output is a text file the Python equilibrium solver reads:
//   H <key> <value>                      header lines
//   I <player> <infoset> <parent_seq> <n> <action>...   one per visit
//   L <prob> <seq_mover> <seq_opp> <spread> <exact>     one per leaf
// <spread> is the final spread from the root mover's point of view: exact
// endgame spread, or with -window a class (-1 loss, 0 tie, 1 win) found by a
// [-1,1]-window search. <exact> is 0 for leaves scored by a greedy playout
// because the turn limit was reached with tiles still in the bag.
#include "../src/compat/ctime.h"
#include "../src/compat/memory_info.h"
#include "../src/def/board_defs.h"
#include "../src/def/game_defs.h"
#include "../src/def/letter_distribution_defs.h"
#include "../src/def/move_defs.h"
#include "../src/def/peg_defs.h"
#include "../src/ent/bag.h"
#include "../src/ent/board.h"
#include "../src/ent/endgame_results.h"
#include "../src/ent/equity.h"
#include "../src/ent/game.h"
#include "../src/ent/letter_distribution.h"
#include "../src/ent/move.h"
#include "../src/ent/player.h"
#include "../src/ent/rack.h"
#include "../src/ent/thread_control.h"
#include "../src/ent/transposition_table.h"
#include "../src/ent/validated_move.h"
#include "../src/impl/config.h"
#include "../src/impl/endgame.h"
#include "../src/impl/gameplay.h"
#include "../src/impl/move_gen.h"
#include "../src/str/move_string.h"
#include "../src/util/io_util.h"
#include "../src/util/string_util.h"
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXL MAX_ALPHABET_SIZE
#define MAX_DEPTH 12
#define KEYLEN 1024
#define MAX_CANDS 256

typedef struct {
  uint8_t c[MAXL];
} Counts;

typedef struct {
  const char *cgp;
  const char *forced;      // comma-separated UCGI moves, used when legal
  int keep[2], empty[2];   // per player (0 = root mover, 1 = opponent)
  int deep_keep, deep_empty; // widths after each side's first decision
  bool center_opp;          // level-k closure around the opponent's rack
  bool allow_pass;
  int level;               // 0 = all deals
  int maxturns;            // pre-endgame turns before a greedy playout leaf
  int plies;               // endgame search depth
  bool window;             // W/T/L classes only
  int shard, nshards;
  const char *out;
  const char *data_paths;
} Opts;

static Opts opt;
static Config *config;
static Game *base;
static const LetterDistribution *ld;
static int ld_size;
static int mover0;
static Counts pool;
static int pool_n;
static FILE *out;
static Game *scratch[MAX_DEPTH + 2];
static MoveList *gen_list;
static EndgameCtx *eg_ctx;
static EndgameResults *eg_results;
static TranspositionTable *eg_tt;
static long n_leaves, n_endgames, n_cache_hits, n_greedy;

// ---------------------------------------------------------------- utilities
static void die(const char *msg) {
  fprintf(stderr, "pegeq: %s\n", msg);
  exit(1);
}

static void counts_from_rack(const Rack *r, Counts *c) {
  memset(c, 0, sizeof(*c));
  for (int ml = 0; ml < ld_size; ml++) {
    c->c[ml] = (uint8_t)rack_get_letter(r, ml);
  }
}

static int counts_total(const Counts *c) {
  int t = 0;
  for (int ml = 0; ml < ld_size; ml++) {
    t += c->c[ml];
  }
  return t;
}

static void counts_str(const Counts *c, char *buf) {
  int n = 0;
  for (int ml = 0; ml < ld_size; ml++) {
    for (int i = 0; i < c->c[ml]; i++) {
      char *hl = ld_ml_to_hl(ld, (MachineLetter)ml);
      buf[n++] = (hl[0] == '\0' || ml == BLANK_MACHINE_LETTER) ? '?' : hl[0];
      free(hl);
    }
  }
  if (n == 0) {
    buf[n++] = '-';
  }
  buf[n] = '\0';
}

static double lchoose(int n, int k) {
  if (k < 0 || k > n) {
    return -INFINITY;
  }
  return lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0);
}

// enumerate all sub-multisets of `avail` with exactly k tiles
typedef void (*SubCb)(const Counts *sub, void *ud);
static void enum_sub(const Counts *avail, int k, int ml, Counts *cur, SubCb cb,
                     void *ud) {
  if (k == 0) {
    cb(cur, ud);
    return;
  }
  if (ml >= ld_size) {
    return;
  }
  int rest = 0;
  for (int j = ml; j < ld_size; j++) {
    rest += avail->c[j];
  }
  if (rest < k) {
    return;
  }
  for (int n = (avail->c[ml] < k ? avail->c[ml] : k); n >= 0; n--) {
    cur->c[ml] = (uint8_t)n;
    enum_sub(avail, k - n, ml + 1, cur, cb, ud);
  }
  cur->c[ml] = 0;
}

static void counts_sub(const Counts *a, const Counts *b, Counts *r) {
  for (int ml = 0; ml < MAXL; ml++) {
    r->c[ml] = (uint8_t)(a->c[ml] - b->c[ml]);
  }
}

// ---------------------------------------------------------------- deal set
typedef struct {
  Counts m, o;
  double logw;
} Deal;

static Deal *deals;
static int n_deals, cap_deals;

static int counts_cmp(const Counts *a, const Counts *b) {
  return memcmp(a->c, b->c, (size_t)ld_size);
}

static int deal_find(const Counts *m, const Counts *o) {
  for (int i = 0; i < n_deals; i++) {
    if (counts_cmp(&deals[i].m, m) == 0 && counts_cmp(&deals[i].o, o) == 0) {
      return i;
    }
  }
  return -1;
}

static void deal_add(const Counts *m, const Counts *o) {
  if (deal_find(m, o) >= 0) {
    return;
  }
  if (n_deals == cap_deals) {
    cap_deals = cap_deals ? cap_deals * 2 : 1024;
    deals = realloc(deals, sizeof(Deal) * (size_t)cap_deals);
  }
  Deal *d = &deals[n_deals++];
  d->m = *m;
  d->o = *o;
  d->logw = 0;
  for (int ml = 0; ml < ld_size; ml++) {
    d->logw += lchoose(pool.c[ml], m->c[ml]) +
               lchoose(pool.c[ml] - m->c[ml], o->c[ml]);
  }
}

typedef struct {
  Counts *list;
  int n, cap;
} RackSet;

static void rackset_add(RackSet *s, const Counts *c) {
  for (int i = 0; i < s->n; i++) {
    if (counts_cmp(&s->list[i], c) == 0) {
      return;
    }
  }
  if (s->n == s->cap) {
    s->cap = s->cap ? s->cap * 2 : 64;
    s->list = realloc(s->list, sizeof(Counts) * (size_t)s->cap);
  }
  s->list[s->n++] = *c;
}

typedef struct {
  const Counts *fixed;
  bool fixed_is_mover;
  RackSet *found;
} PairCtx;

static void pair_cb(const Counts *sub, void *ud) {
  PairCtx *p = ud;
  if (p->fixed_is_mover) {
    deal_add(p->fixed, sub);
  } else {
    deal_add(sub, p->fixed);
  }
  rackset_add(p->found, sub);
}

static void collect_cb(const Counts *sub, void *ud) { rackset_add(ud, sub); }

static void build_deals(const Counts *r0, const Counts *o0, int rack_n_m,
                        int rack_n_o) {
  RackSet cur_m = {0}, cur_o = {0};
  if (opt.center_opp) {
    rackset_add(&cur_o, o0);
  } else {
    rackset_add(&cur_m, r0);
  }
  if (opt.level <= 0) {
    // every deal: all mover racks, each with all opponent racks
    RackSet all_m = {0};
    Counts tmp = {0};
    enum_sub(&pool, rack_n_m, 0, &tmp, collect_cb, &all_m);
    for (int i = 0; i < all_m.n; i++) {
      Counts rest;
      counts_sub(&pool, &all_m.list[i], &rest);
      RackSet dummy = {0};
      PairCtx p2 = {.fixed = &all_m.list[i], .fixed_is_mover = true,
                    .found = &dummy};
      Counts t2 = {0};
      enum_sub(&rest, rack_n_o, 0, &t2, pair_cb, &p2);
      free(dummy.list);
    }
    free(all_m.list);
    return;
  }
  RackSet seen_m = {0}, seen_o = {0};
  if (opt.center_opp) {
    rackset_add(&seen_o, o0);
  } else {
    rackset_add(&seen_m, r0);
  }
  for (int lev = 1; lev <= opt.level; lev++) {
    RackSet next = {0};
    if ((lev % 2 == 1) != opt.center_opp) {
      for (int i = 0; i < cur_m.n; i++) {
        Counts rest, t = {0};
        counts_sub(&pool, &cur_m.list[i], &rest);
        PairCtx p = {.fixed = &cur_m.list[i], .fixed_is_mover = true,
                     .found = &next};
        enum_sub(&rest, rack_n_o, 0, &t, pair_cb, &p);
      }
      cur_o.n = 0;
      for (int i = 0; i < next.n; i++) {
        int before = seen_o.n;
        rackset_add(&seen_o, &next.list[i]);
        if (seen_o.n > before) {
          rackset_add(&cur_o, &next.list[i]);
        }
      }
    } else {
      for (int i = 0; i < cur_o.n; i++) {
        Counts rest, t = {0};
        counts_sub(&pool, &cur_o.list[i], &rest);
        PairCtx p = {.fixed = &cur_o.list[i], .fixed_is_mover = false,
                     .found = &next};
        enum_sub(&rest, rack_n_m, 0, &t, pair_cb, &p);
      }
      cur_m.n = 0;
      for (int i = 0; i < next.n; i++) {
        int before = seen_m.n;
        rackset_add(&seen_m, &next.list[i]);
        if (seen_m.n > before) {
          rackset_add(&cur_m, &next.list[i]);
        }
      }
    }
    free(next.list);
    fprintf(stderr, "level %d: %d deals\n", lev, n_deals);
  }
}

// ---------------------------------------------------------------- game state
static void set_rack(Game *g, int p, const Counts *c) {
  Rack *r = player_get_rack(game_get_player(g, p));
  rack_reset(r);
  for (int ml = 0; ml < ld_size; ml++) {
    for (int i = 0; i < c->c[ml]; i++) {
      rack_add_letter(r, (MachineLetter)ml);
    }
  }
}

static void set_bag(Game *g, const Counts *c) {
  MachineLetter tiles[MAX_BAG_SIZE];
  int n = 0;
  for (int ml = 0; ml < ld_size; ml++) {
    for (int i = 0; i < c->c[ml]; i++) {
      tiles[n++] = (MachineLetter)ml;
    }
  }
  bag_set_to_tiles(game_get_bag(g), tiles, n);
}

static void bag_counts(const Game *g, Counts *c) {
  MachineLetter tiles[MAX_BAG_SIZE];
  const int n = bag_peek_tiles(game_get_bag(g), tiles);
  memset(c, 0, sizeof(*c));
  for (int i = 0; i < n; i++) {
    c->c[tiles[i]]++;
  }
}

static int rack_points(const Game *g, int p) {
  return equity_to_int(
      rack_get_score(ld, player_get_rack(game_get_player(g, p))));
}

static int score_of(const Game *g, int p) {
  return equity_to_int(player_get_score(game_get_player(g, p)));
}

// ---------------------------------------------------------------- leaf values
typedef struct {
  uint64_t key;
  int32_t val;
  bool used;
} CacheEntry;
static CacheEntry *cache;
static size_t cache_cap = 1u << 22;

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
  const uint8_t *b = p;
  for (size_t i = 0; i < n; i++) {
    h ^= b[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static uint64_t state_key(const Game *g) {
  uint64_t h = 1469598103934665603ULL;
  const Board *b = game_get_board(g);
  for (int r = 0; r < BOARD_DIM; r++) {
    for (int c = 0; c < BOARD_DIM; c++) {
      MachineLetter ml = board_is_empty(b, r, c) ? 255 : board_get_letter(b, r, c);
      h = fnv(h, &ml, 1);
    }
  }
  for (int p = 0; p < 2; p++) {
    Counts c;
    counts_from_rack(player_get_rack(game_get_player(g, p)), &c);
    h = fnv(h, c.c, (size_t)ld_size);
    int s = score_of(g, p);
    h = fnv(h, &s, sizeof(s));
  }
  int t = game_get_player_on_turn_index(g);
  int z = game_get_consecutive_scoreless_turns(g);
  h = fnv(h, &t, sizeof(t));
  h = fnv(h, &z, sizeof(z));
  h = fnv(h, &opt.window, 1);
  return h;
}

// Final spread from the root mover's point of view (or its sign with -window).
static int32_t finish_value(int32_t spread_for_mover0) {
  if (!opt.window) {
    return spread_for_mover0;
  }
  return spread_for_mover0 > 0 ? 1 : (spread_for_mover0 < 0 ? -1 : 0);
}

static int32_t game_over_value(const Game *g) {
  // Six scoreless turns: each player loses their own rack value.
  int s0 = score_of(g, mover0) - rack_points(g, mover0);
  int s1 = score_of(g, 1 - mover0) - rack_points(g, 1 - mover0);
  return finish_value(s0 - s1);
}

static int32_t endgame_value(Game *g) {
  if (game_get_game_end_reason(g) != GAME_END_REASON_NONE) {
    return game_over_value(g);
  }
  const uint64_t key = state_key(g);
  size_t slot = key & (cache_cap - 1);
  while (cache[slot].used) {
    if (cache[slot].key == key) {
      n_cache_hits++;
      return cache[slot].val;
    }
    slot = (slot + 1) & (cache_cap - 1);
  }
  const int on_turn = game_get_player_on_turn_index(g);
  EndgameArgs ea;
  endgame_args_fill(
      config_get_thread_control(config), g, 0.0, opt.plies,
      DEFAULT_INITIAL_SMALL_MOVE_ARENA_SIZE, 1, true, 1, NULL, NULL, NULL,
      NULL, NULL, NULL, DUAL_LEXICON_MODE_IGNORANT, true, true, false, false,
      0.0, 0.0, PEG_ENDGAME_SEED, false, eg_tt, 0, opt.window, -1, false, 0, 0,
      0, NULL, &ea);
  endgame_results_reset(eg_results);
  endgame_solve_inline(&eg_ctx, &ea, eg_results);
  n_endgames++;
  if (endgame_results_get_depth(eg_results, ENDGAME_RESULT_BEST) < 0) {
    die("endgame solve returned no result");
  }
  const int32_t eg_val =
      endgame_results_get_value(eg_results, ENDGAME_RESULT_BEST);
  const int32_t lead = score_of(g, on_turn) - score_of(g, 1 - on_turn);
  int32_t spread_on_turn = lead + eg_val;
  int32_t v = on_turn == mover0 ? spread_on_turn : -spread_on_turn;
  if (opt.window) {
    // first_win: eg_val is only reliable in sign relative to the window.
    v = v > 0 ? 1 : (v < 0 ? -1 : 0);
  }
  cache[slot].used = true;
  cache[slot].key = key;
  cache[slot].val = v;
  return v;
}

// Greedy playout to the end of the game: each side plays its top-equity move
// and draws from the bag in its current order. Used only past -maxturns.
static int32_t greedy_value(Game *g, int depth) {
  Game *w = scratch[MAX_DEPTH + 1];
  game_copy(w, g);
  MoveList *ml = move_list_create(1);
  int guard = 0;
  while (game_get_game_end_reason(w) == GAME_END_REASON_NONE && guard++ < 60) {
    const Move *m = get_top_equity_move(w, ml);
    play_move(m, w, NULL);
  }
  move_list_destroy(ml);
  (void)depth;
  n_greedy++;
  int s0 = score_of(w, mover0), s1 = score_of(w, 1 - mover0);
  return finish_value(s0 - s1);
}

// ---------------------------------------------------------------- candidates
typedef struct {
  Move moves[MAX_CANDS];
  char names[MAX_CANDS][48];
  int n;
} Cands;

static void move_name(const Game *g, const Move *m, char *buf) {
  StringBuilder *sb = string_builder_create();
  string_builder_add_move(sb, game_get_board(g), m, ld, false);
  snprintf(buf, 48, "%s", string_builder_peek(sb));
  for (char *p = buf; *p; p++) {
    if (*p == ' ') {
      *p = '_';
    }
  }
  string_builder_destroy(sb);
}

static void cands_add(Cands *cs, const Game *g, const Move *m) {
  char name[48];
  move_name(g, m, name);
  for (int i = 0; i < cs->n; i++) {
    if (strcmp(cs->names[i], name) == 0) {
      return;
    }
  }
  if (cs->n >= MAX_CANDS) {
    return;
  }
  cs->moves[cs->n] = *m;
  strcpy(cs->names[cs->n], name);
  cs->n++;
}

static int cmp_move_eq(const void *a, const void *b) {
  const Move *x = *(const Move *const *)a, *y = *(const Move *const *)b;
  Equity ex = move_get_equity(x), ey = move_get_equity(y);
  return ex < ey ? 1 : (ex > ey ? -1 : 0);
}

static void get_cands(const Game *g, Cands *cs, int depth) {
  cs->n = 0;
  const int p = game_get_player_on_turn_index(g);
  const int role = p == mover0 ? 0 : 1;
  const int bag = bag_get_letters(game_get_bag(g));
  // forced moves first (kept only when legal for this rack)
  if (opt.forced != NULL && opt.forced[0] != '\0') {
    char *copy = string_duplicate(opt.forced);
    for (char *tok = strtok(copy, ","); tok != NULL; tok = strtok(NULL, ",")) {
      ErrorStack *es = error_stack_create();
      ValidatedMoves *vm = validated_moves_create(g, p, tok, false, true, es);
      if (error_stack_is_empty(es) && vm != NULL &&
          validated_moves_get_number_of_moves(vm) > 0) {
        const Move *m = validated_moves_get_move(vm, 0);
        if (move_get_type(m) == GAME_EVENT_TILE_PLACEMENT_MOVE ||
            (move_get_type(m) == GAME_EVENT_PASS && opt.allow_pass)) {
          cands_add(cs, g, m);
        }
      }
      if (vm != NULL) {
        validated_moves_destroy(vm);
      }
      error_stack_destroy(es);
    }
    free(copy);
  }
  const MoveGenArgs ga = {
      .game = g,
      .move_record_type = MOVE_RECORD_ALL,
      .move_sort_type = MOVE_SORT_EQUITY,
      .override_kwg = NULL,
      .eq_margin_movegen = 0,
      .target_equity = EQUITY_MAX_VALUE,
      .move_list = gen_list,
      .tiles_played_bv = NULL,
      .initial_tiles_bv = 0,
  };
  generate_moves(&ga);
  const int n = move_list_get_count(gen_list);
  const Move **sorted = malloc(sizeof(Move *) * (size_t)(n + 1));
  int ns = 0;
  for (int i = 0; i < n; i++) {
    const Move *m = move_list_get_move(gen_list, i);
    if (move_get_type(m) == GAME_EVENT_TILE_PLACEMENT_MOVE) {
      sorted[ns++] = m;
    }
  }
  qsort(sorted, (size_t)ns, sizeof(Move *), cmp_move_eq);
  int k_keep = 0, k_empty = 0;
  for (int i = 0; i < ns; i++) {
    const bool empties = move_get_tiles_played(sorted[i]) >= bag;
    const int max_empty = depth < 2 ? opt.empty[role] : opt.deep_empty;
    const int max_keep = depth < 2 ? opt.keep[role] : opt.deep_keep;
    if (empties && k_empty < max_empty) {
      cands_add(cs, g, sorted[i]);
      k_empty++;
    } else if (!empties && k_keep < max_keep) {
      cands_add(cs, g, sorted[i]);
      k_keep++;
    }
  }
  free(sorted);
  if (opt.allow_pass || cs->n == 0) {
    Move pass;
    memset(&pass, 0, sizeof(pass));
    move_set_as_pass(&pass);
    cands_add(cs, g, &pass);
  }
}

// ---------------------------------------------------------------- tree walk
typedef struct {
  double prob;
  int depth;
  char priv[2][KEYLEN];
  char pub[KEYLEN];
  char seq[2][KEYLEN];
} Path;

static void walk(Game *g, Path *path);

typedef struct {
  Game *after;      // position after the play, before drawing
  Path *path;
  Counts bag;
  int bagn, k, player;
  double logtot;
} DrawCtx;

static void draw_cb(const Counts *d, void *ud) {
  DrawCtx *dc = ud;
  double lp = -dc->logtot;
  for (int ml = 0; ml < ld_size; ml++) {
    lp += lchoose(dc->bag.c[ml], d->c[ml]);
  }
  Game *g = scratch[dc->path->depth + 1];
  game_copy(g, dc->after);
  Counts rest;
  counts_sub(&dc->bag, d, &rest);
  set_bag(g, &rest);
  Rack *r = player_get_rack(game_get_player(g, dc->player));
  for (int ml = 0; ml < ld_size; ml++) {
    for (int i = 0; i < d->c[ml]; i++) {
      rack_add_letter(r, (MachineLetter)ml);
    }
  }
  Path np = *dc->path;
  np.depth++;
  np.prob *= exp(lp);
  char ds[32];
  counts_str(d, ds);
  size_t len = strlen(np.priv[dc->player]);
  snprintf(np.priv[dc->player] + len, KEYLEN - len, "+%s", ds);
  walk(g, &np);
}

static void walk(Game *g, Path *path) {
  if (path->depth > MAX_DEPTH) {
    die("tree too deep");
  }
  const int bag = bag_get_letters(game_get_bag(g));
  if (game_get_game_end_reason(g) != GAME_END_REASON_NONE || bag == 0 ||
      path->depth >= opt.maxturns * 2) {
    int32_t v;
    int exact = 1;
    if (game_get_game_end_reason(g) != GAME_END_REASON_NONE) {
      v = game_over_value(g);
    } else if (bag == 0) {
      v = endgame_value(g);
    } else {
      v = greedy_value(g, path->depth);
      exact = 0;
    }
    n_leaves++;
    fprintf(out, "L %.17g %s %s %d %d\n", path->prob, path->seq[0],
            path->seq[1], v, exact);
    return;
  }
  const int p = game_get_player_on_turn_index(g);
  const int role = p == mover0 ? 0 : 1;
  Cands cs;
  get_cands(g, &cs, path->depth);
  char infoset[KEYLEN * 2 + 8];
  snprintf(infoset, sizeof(infoset), "%d|%s|%s", role, path->priv[role],
           path->pub);
  fprintf(out, "I %d %s %s %d", role, infoset, path->seq[role], cs.n);
  for (int i = 0; i < cs.n; i++) {
    fprintf(out, " %s", cs.names[i]);
  }
  fprintf(out, "\n");
  Game *after = scratch[path->depth + 1];
  Counts bagc;
  bag_counts(g, &bagc);
  for (int i = 0; i < cs.n; i++) {
    Path np = *path;
    snprintf(np.seq[role], KEYLEN, "%s#%s", infoset, cs.names[i]);
    size_t len = strlen(np.pub);
    snprintf(np.pub + len, KEYLEN - len, "%s%s", len ? ";" : "", cs.names[i]);
    const Move *m = &cs.moves[i];
    const int t = move_get_type(m) == GAME_EVENT_TILE_PLACEMENT_MOVE
                      ? move_get_tiles_played(m)
                      : 0;
    const int k = t < bag ? t : bag;
    // play without drawing on a private copy, then branch on the draw
    Game *pg = game_duplicate(g);
    play_move_without_drawing_tiles(m, pg);
    if (k == 0) {
      np.depth++;
      game_copy(after, pg);
      walk(after, &np);
    } else {
      DrawCtx dc = {.after = pg, .path = &np, .bag = bagc, .bagn = bag,
                    .k = k, .player = p, .logtot = lchoose(bag, k)};
      Counts tmp = {0};
      enum_sub(&bagc, k, 0, &tmp, draw_cb, &dc);
    }
    game_destroy(pg);
  }
}

// ---------------------------------------------------------------- main
static int parse_int(const char *s) { return (int)strtol(s, NULL, 10); }

int main(int argc, char **argv) {
  memset(&opt, 0, sizeof(opt));
  opt.keep[0] = opt.keep[1] = 6;
  opt.empty[0] = opt.empty[1] = 6;
  opt.allow_pass = true;
  opt.deep_keep = 2;
  opt.deep_empty = 4;
  opt.level = 2;
  opt.maxturns = 3;
  opt.plies = 25;
  opt.nshards = 1;
  opt.out = "pegeq_tree.txt";
  opt.data_paths = "./data";
  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
    const char *v = i + 1 < argc ? argv[i + 1] : "";
    if (!strcmp(a, "-cgp")) { opt.cgp = v; i++; }
    else if (!strcmp(a, "-forced")) { opt.forced = v; i++; }
    else if (!strcmp(a, "-keep")) { opt.keep[0] = opt.keep[1] = parse_int(v); i++; }
    else if (!strcmp(a, "-empty")) { opt.empty[0] = opt.empty[1] = parse_int(v); i++; }
    else if (!strcmp(a, "-mkeep")) { opt.keep[0] = parse_int(v); i++; }
    else if (!strcmp(a, "-okeep")) { opt.keep[1] = parse_int(v); i++; }
    else if (!strcmp(a, "-mempty")) { opt.empty[0] = parse_int(v); i++; }
    else if (!strcmp(a, "-oempty")) { opt.empty[1] = parse_int(v); i++; }
    else if (!strcmp(a, "-nopass")) { opt.allow_pass = false; }
    else if (!strcmp(a, "-center")) { opt.center_opp = !strcmp(v, "opp"); i++; }
    else if (!strcmp(a, "-deepkeep")) { opt.deep_keep = parse_int(v); i++; }
    else if (!strcmp(a, "-deepempty")) { opt.deep_empty = parse_int(v); i++; }
    else if (!strcmp(a, "-level")) { opt.level = parse_int(v); i++; }
    else if (!strcmp(a, "-maxturns")) { opt.maxturns = parse_int(v); i++; }
    else if (!strcmp(a, "-plies")) { opt.plies = parse_int(v); i++; }
    else if (!strcmp(a, "-window")) { opt.window = true; }
    else if (!strcmp(a, "-shard")) { opt.shard = parse_int(v); i++; }
    else if (!strcmp(a, "-nshards")) { opt.nshards = parse_int(v); i++; }
    else if (!strcmp(a, "-out")) { opt.out = v; i++; }
    else { fprintf(stderr, "unknown option %s\n", a); return 2; }
  }
  if (opt.cgp == NULL) {
    die("need -cgp '<cgp string>'");
  }
  ErrorStack *es = error_stack_create();
  ConfigArgs ca = {.data_paths = opt.data_paths, .settings_filename = NULL,
                   .use_wmp = true};
  config = config_create(&ca, es);
  if (!error_stack_is_empty(es)) {
    error_stack_print_and_reset(es);
    return 1;
  }
  thread_control_set_status(config_get_thread_control(config),
                            THREAD_CONTROL_STATUS_STARTED);
  char cmd[4096];
  snprintf(cmd, sizeof(cmd), "set -threads 1 -savesettings false");
  config_load_command(config, cmd, es);
  config_execute_command(config, es);
  snprintf(cmd, sizeof(cmd), "cgp %s", opt.cgp);
  config_load_command(config, cmd, es);
  config_execute_command(config, es);
  if (!error_stack_is_empty(es)) {
    error_stack_print_and_reset(es);
    return 1;
  }
  base = config_get_game(config);
  ld = game_get_ld(base);
  ld_size = ld_get_size(ld);
  mover0 = game_get_player_on_turn_index(base);

  // pool = full distribution minus board tiles
  memset(&pool, 0, sizeof(pool));
  for (int ml = 0; ml < ld_size; ml++) {
    pool.c[ml] = (uint8_t)ld_get_dist(ld, ml);
  }
  const Board *b = game_get_board(base);
  for (int r = 0; r < BOARD_DIM; r++) {
    for (int c = 0; c < BOARD_DIM; c++) {
      if (!board_is_empty(b, r, c)) {
        MachineLetter ml = board_get_letter(b, r, c);
        ml = get_is_blanked(ml) ? BLANK_MACHINE_LETTER : ml;
        pool.c[ml]--;
      }
    }
  }
  pool_n = counts_total(&pool);
  Counts r0, o0;
  counts_from_rack(player_get_rack(game_get_player(base, mover0)), &r0);
  counts_from_rack(player_get_rack(game_get_player(base, 1 - mover0)), &o0);
  const int nm = counts_total(&r0), no = counts_total(&o0);
  const int bag0 = pool_n - nm - no;
  char s1[64], s2[64], s3[64];
  counts_str(&r0, s1);
  counts_str(&o0, s2);
  counts_str(&pool, s3);
  fprintf(stderr, "mover rack %s, opp rack %s, pool %s (%d), bag %d\n", s1, s2,
          s3, pool_n, bag0);
  if (bag0 < 1 || bag0 > 6) {
    die("bag must hold 1-6 tiles");
  }
  build_deals(&r0, &o0, nm, no);
  fprintf(stderr, "%d deals\n", n_deals);

  for (int i = 0; i <= MAX_DEPTH + 1; i++) {
    scratch[i] = game_duplicate(base);
  }
  gen_list = move_list_create(PEG_NEST_CAND_LIST_CAP);
  eg_results = endgame_results_create();
  eg_tt = transposition_table_create_with_threads(0.25, PEG_TT_MIN_SIZE_POWER, 1);
  cache = calloc(cache_cap, sizeof(CacheEntry));

  out = fopen(opt.out, "w");
  if (out == NULL) {
    die("cannot open output");
  }
  double logw_max = -INFINITY, wsum = 0;
  for (int i = 0; i < n_deals; i++) {
    if (deals[i].logw > logw_max) {
      logw_max = deals[i].logw;
    }
  }
  for (int i = 0; i < n_deals; i++) {
    wsum += exp(deals[i].logw - logw_max);
  }
  fprintf(out, "H mover_rack %s\nH opp_rack %s\nH pool %s\nH deals %d\n", s1,
          s2, s3, n_deals);
  fprintf(out, "H window %d\nH level %d\n", opt.window, opt.level);
  const int64_t t0 = ctimer_monotonic_ns();
  for (int i = 0; i < n_deals; i++) {
    if (i % opt.nshards != opt.shard) {
      continue;
    }
    Game *g = scratch[0];
    game_copy(g, base);
    set_rack(g, mover0, &deals[i].m);
    set_rack(g, 1 - mover0, &deals[i].o);
    Counts bagc;
    counts_sub(&pool, &deals[i].m, &bagc);
    counts_sub(&bagc, &deals[i].o, &bagc);
    set_bag(g, &bagc);
    Path path;
    memset(&path, 0, sizeof(path));
    path.prob = exp(deals[i].logw - logw_max) / wsum;
    counts_str(&deals[i].m, path.priv[0]);
    counts_str(&deals[i].o, path.priv[1]);
    strcpy(path.seq[0], "root");
    strcpy(path.seq[1], "root");
    walk(g, &path);
    if (i % 50 == 0) {
      fprintf(stderr,
              "deal %d/%d  leaves %ld  endgames %ld  cache hits %ld  "
              "greedy %ld  %.0fs\n",
              i, n_deals, n_leaves, n_endgames, n_cache_hits, n_greedy,
              (double)(ctimer_monotonic_ns() - t0) / 1e9);
    }
  }
  fclose(out);
  fprintf(stderr, "done: leaves %ld endgames %ld cache hits %ld greedy %ld %.0fs\n",
          n_leaves, n_endgames, n_cache_hits, n_greedy,
          (double)(ctimer_monotonic_ns() - t0) / 1e9);
  return 0;
}
