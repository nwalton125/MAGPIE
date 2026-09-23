// Antistatic endgame experiment: how much does solving endgames against a
// static opponent gain over playing them statically, and over solving them
// normally?
//
// For each seed, both players play their top static equity move until the bag
// is empty. The endgame is then played out five ways from that same position:
// static vs static (the baseline), then each solver as player one against a
// static player two and as player two against a static player one.
// Everything before the endgame is shared, so any difference in the result
// comes from the endgame.
//
// The two solvers run the endgame solver on each of their turns with the same
// time limit and search mode:
//   - antistatic: opponent_static set, so its opponent is modeled exactly as
//     the static player it faces;
//   - normal: the endgame command's solver, which assumes the opponent plays
//     its best reply.
// Each solver has its own transposition table, since a position's value
// differs between the two opponent models.
//
// Environment variables (all optional):
//   ANTISTATIC_GAMES     number of games (default 10)
//   ANTISTATIC_SEED      seed of the first game; game i uses seed + i
//                        (default 1)
//   ANTISTATIC_SECONDS   per-move endgame solver time limit, for both solvers
//                        (default 10). The
//                        solver searches until this deadline or until it
//                        completes; a solve cut short plays the best move from
//                        its deepest completed depth and is counted as
//                        incomplete. If the solver returns no move at all, the
//                        static move is played and counted.
//   ANTISTATIC_FIRSTWIN  1 = first-win search (default), 0 = maximize spread;
//                        applies to both solvers
//   ANTISTATIC_LEX       lexicon (default NWL23)
//   ANTISTATIC_OUT       directory for the per-game logs (default
//                        antistatic_games); a summary is printed to stdout

#include "antistatic_endgame_test.h"

#include "../src/compat/ctime.h"
#include "../src/def/game_defs.h"
#include "../src/def/thread_control_defs.h"
#include "../src/ent/bag.h"
#include "../src/ent/endgame_results.h"
#include "../src/ent/equity.h"
#include "../src/ent/game.h"
#include "../src/ent/letter_distribution.h"
#include "../src/ent/move.h"
#include "../src/ent/player.h"
#include "../src/ent/thread_control.h"
#include "../src/impl/cgp.h"
#include "../src/impl/config.h"
#include "../src/impl/endgame.h"
#include "../src/impl/gameplay.h"
#include "../src/str/move_string.h"
#include "../src/str/rack_string.h"
#include "../src/util/io_util.h"
#include "../src/util/string_util.h"
#include "test_constants.h"
#include "test_util.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

enum {
  NO_SOLVER_PLAYER = -1,
  // Variants: static vs static, then each solver as player one and as player
  // two.
  NUM_VARIANTS = 5,
};

typedef enum {
  SOLVER_ANTISTATIC,
  SOLVER_NORMAL,
  NUM_SOLVERS,
} solver_t;

static const char *const solver_names[NUM_SOLVERS] = {"Antistatic", "Normal"};

typedef struct Variant {
  const char *name;
  int solver_player; // NO_SOLVER_PLAYER for static vs static
  solver_t solver;
} Variant;

static const Variant variants[NUM_VARIANTS] = {
    {"Static vs static", NO_SOLVER_PLAYER, SOLVER_ANTISTATIC},
    {"Antistatic P1 vs static P2", 0, SOLVER_ANTISTATIC},
    {"Static P1 vs antistatic P2", 1, SOLVER_ANTISTATIC},
    {"Normal solver P1 vs static P2", 0, SOLVER_NORMAL},
    {"Static P1 vs normal solver P2", 1, SOLVER_NORMAL},
};

typedef struct SolverStats {
  int moves;
  // Solves cut short by the time limit before completing the full depth.
  int incomplete;
  // Solves that returned no move, so the static move was played instead.
  int no_move;
  double seconds;
} SolverStats;

typedef struct EndgameOutcome {
  int final_spread; // player one's score minus player two's
  SolverStats stats;
} EndgameOutcome;

static int env_int(const char *name, int default_value) {
  const char *value = getenv(name);
  return value ? (int)strtol(value, NULL, 10) : default_value;
}

static double env_double(const char *name, double default_value) {
  const char *value = getenv(name);
  return value ? strtod(value, NULL) : default_value;
}

static const char *env_string(const char *name, const char *default_value) {
  const char *value = getenv(name);
  return value ? value : default_value;
}

static int player_one_spread(const Game *game) {
  return equity_to_int(player_get_score(game_get_player(game, 0)) -
                       player_get_score(game_get_player(game, 1)));
}

static int result_sign(int spread) { return (spread > 0) - (spread < 0); }

static const char *result_string(int spread) {
  if (spread > 0) {
    return "win";
  }
  if (spread < 0) {
    return "loss";
  }
  return "draw";
}

// Appends "P<n> <rack>  <move>  <score1>-<score2>" for a move about to be
// played, then plays it.
static void log_and_play_move(StringBuilder *log, const Move *move, Game *game,
                              const char *annotation) {
  const LetterDistribution *ld = game_get_ld(game);
  const int on_turn = game_get_player_on_turn_index(game);
  string_builder_add_formatted_string(log, "  P%d ", on_turn + 1);
  string_builder_add_rack(log, player_get_rack(game_get_player(game, on_turn)),
                          ld, false);
  string_builder_add_string(log, "  ");
  string_builder_add_move(log, game_get_board(game), move, ld, true);
  play_move(move, game, NULL);
  string_builder_add_formatted_string(
      log, "  %d-%d%s\n",
      equity_to_int(player_get_score(game_get_player(game, 0))),
      equity_to_int(player_get_score(game_get_player(game, 1))), annotation);
}

// Picks a move with the endgame solver. Returns false if the solver produced
// no move. Sets *depth to the depth the chosen move was searched to.
static bool choose_solver_move(Game *game, solver_t solver, EndgameCtx **ctx,
                               EndgameResults *results, double seconds,
                               bool first_win, Move *out_move, int *depth,
                               double *elapsed) {
  ThreadControl *thread_control = thread_control_create();
  thread_control_set_status(thread_control, THREAD_CONTROL_STATUS_STARTED);
  EndgameArgs args = {0};
  args.thread_control = thread_control;
  args.game = game;
  args.plies = MAX_VARIANT_LENGTH;
  args.tt_fraction_of_mem = 0.05;
  args.initial_small_move_arena_size = DEFAULT_INITIAL_SMALL_MOVE_ARENA_SIZE;
  args.num_threads = 1;
  // The endgame command's defaults. opponent_static turns off both.
  args.use_heuristics = true;
  args.incremental_movegen = true;
  args.forced_pass_bypass = true;
  args.num_top_moves = 1;
  // No soft or hard limit: those stop iterative deepening early when the next
  // depth is predicted not to fit, to save time for later moves. Here there
  // is no clock to save time for, so search until the deadline.
  args.soft_time_limit = 0.0;
  args.hard_time_limit = 0.0;
  args.external_deadline_ns =
      ctimer_monotonic_ns() + (int64_t)(seconds * 1.0e9);
  args.seed = 42;
  args.first_win = first_win;
  args.opponent_static = solver == SOLVER_ANTISTATIC;

  Timer timer;
  ctimer_start(&timer);
  ErrorStack *error_stack = error_stack_create();
  endgame_solve(ctx, &args, results, error_stack);
  *elapsed = ctimer_elapsed_seconds(&timer);
  *depth = 0;
  const bool ok = error_stack_is_empty(error_stack);
  error_stack_destroy(error_stack);
  thread_control_destroy(thread_control);
  if (!ok) {
    return false;
  }
  const PVLine *pv_line =
      endgame_results_get_pvline(results, ENDGAME_RESULT_BEST);
  if (pv_line->num_moves == 0) {
    return false;
  }
  small_move_to_move(out_move, &pv_line->moves[0], game_get_board(game));
  *depth = endgame_results_get_depth(results, ENDGAME_RESULT_BEST);
  return true;
}

// Plays out the endgame from `start`: the variant's solver player (if any)
// uses its solver and everyone else plays statically.
static EndgameOutcome
play_out_endgame(const Game *start, const Variant *variant, double seconds,
                 bool first_win, MoveList *move_list, EndgameCtx **ctx,
                 EndgameResults *results, StringBuilder *log) {
  EndgameOutcome outcome = {0};
  Game *game = game_duplicate(start);
  Move move;
  while (game_get_game_end_reason(game) == GAME_END_REASON_NONE) {
    if (game_get_player_on_turn_index(game) == variant->solver_player) {
      int depth = 0;
      double elapsed = 0.0;
      const bool ok =
          choose_solver_move(game, variant->solver, ctx, results, seconds,
                             first_win, &move, &depth, &elapsed);
      outcome.stats.moves++;
      outcome.stats.seconds += elapsed;
      char annotation[96];
      if (ok) {
        const bool incomplete = depth < MAX_VARIANT_LENGTH;
        outcome.stats.incomplete += incomplete;
        (void)snprintf(annotation, sizeof(annotation),
                       "  [solver %.2fs, depth %d%s]", elapsed, depth,
                       incomplete ? ", incomplete" : "");
      } else {
        outcome.stats.no_move++;
        move_copy(&move, get_top_equity_move(game, move_list));
        (void)snprintf(annotation, sizeof(annotation),
                       "  [solver %.2fs, no move; static move played]",
                       elapsed);
      }
      log_and_play_move(log, &move, game, annotation);
    } else {
      move_copy(&move, get_top_equity_move(game, move_list));
      log_and_play_move(log, &move, game, "");
    }
  }
  outcome.final_spread = player_one_spread(game);
  string_builder_add_formatted_string(
      log, "  Final: %d-%d, P1 spread %+d (P1 %s)\n",
      equity_to_int(player_get_score(game_get_player(game, 0))),
      equity_to_int(player_get_score(game_get_player(game, 1))),
      outcome.final_spread, result_string(outcome.final_spread));
  game_destroy(game);
  return outcome;
}

void test_antistatic_endgame_experiment(void) {
  const int num_games = env_int("ANTISTATIC_GAMES", 10);
  const int first_seed = env_int("ANTISTATIC_SEED", 1);
  const double seconds = env_double("ANTISTATIC_SECONDS", 10.0);
  const bool first_win = env_int("ANTISTATIC_FIRSTWIN", 1) != 0;
  const char *lexicon = env_string("ANTISTATIC_LEX", "NWL23");
  const char *out_dir = env_string("ANTISTATIC_OUT", "antistatic_games");
  mkdir(out_dir, 0755);

  char *config_cmd = get_formatted_string(
      "set -lex %s -s1 equity -s2 equity -r1 best -r2 best -threads 1",
      lexicon);
  Config *config = config_create_or_die(config_cmd);
  free(config_cmd);
  char *empty_cgp_cmd = get_formatted_string(
      "cgp %s -lex %s;", EMPTY_CGP_WITHOUT_OPTIONS, lexicon);
  load_and_exec_config_or_die(config, empty_cgp_cmd);
  free(empty_cgp_cmd);
  Game *game = config_get_game(config);
  MoveList *move_list = move_list_create(1);
  EndgameResults *results = endgame_results_create();
  // One context, and so one transposition table, per solver.
  EndgameCtx *ctxs[NUM_SOLVERS] = {NULL, NULL};
  ErrorStack *error_stack = error_stack_create();

  printf("antistatic endgame experiment: %d games from seed %d, %.1fs per "
         "move, %s, lexicon %s\n",
         num_games, first_seed, seconds, first_win ? "first-win" : "max spread",
         lexicon);
  printf("%-6s %-6s %-9s %-9s %-9s %-9s %-9s %-8s %-8s %s\n", "game", "seed",
         "static", "anti-P1", "anti-P2", "norm-P1", "norm-P2", "anti-inc",
         "norm-inc", "no-move");
  (void)fflush(stdout);

  // Outcome changes relative to the static baseline from the solver's seat:
  // [solver][seat].
  int improved[NUM_SOLVERS][2] = {{0}};
  int worsened[NUM_SOLVERS][2] = {{0}};
  SolverStats totals[NUM_SOLVERS] = {{0}};
  int games_played = 0;

  for (int game_index = 0; game_index < num_games; game_index++) {
    const int seed = first_seed + game_index;
    game_reset(game);
    game_seed(game, (uint64_t)seed);
    game_set_starting_player_index(game, 0);
    draw_starting_racks(game);

    StringBuilder *log = string_builder_create();
    string_builder_add_formatted_string(log, "Game %d (seed %d)\n\n",
                                        game_index + 1, seed);
    string_builder_add_string(log, "Static play until the bag is empty:\n");
    while (game_get_game_end_reason(game) == GAME_END_REASON_NONE &&
           !bag_is_empty(game_get_bag(game))) {
      const Move *move = get_top_equity_move(game, move_list);
      Move move_copy_value;
      move_copy(&move_copy_value, move);
      log_and_play_move(log, &move_copy_value, game, "");
    }
    if (game_get_game_end_reason(game) != GAME_END_REASON_NONE) {
      // Ended before the bag emptied (e.g. six scoreless turns): no endgame.
      printf("%-6d %-6d (game ended before the endgame)\n", game_index + 1,
             seed);
      (void)fflush(stdout);
      string_builder_destroy(log);
      continue;
    }
    games_played++;

    char *cgp = game_get_cgp(game, true);
    string_builder_add_formatted_string(
        log, "\nEndgame position (CGP, player on turn first):\n  %s -lex %s\n",
        cgp, lexicon);
    free(cgp);

    EndgameOutcome outcomes[NUM_VARIANTS];
    SolverStats game_stats[NUM_SOLVERS] = {{0}};
    for (int variant_idx = 0; variant_idx < NUM_VARIANTS; variant_idx++) {
      const Variant *variant = &variants[variant_idx];
      string_builder_add_formatted_string(log, "\n%s:\n", variant->name);
      outcomes[variant_idx] =
          play_out_endgame(game, variant, seconds, first_win, move_list,
                           &ctxs[variant->solver], results, log);
      if (variant->solver_player == NO_SOLVER_PLAYER) {
        continue;
      }
      const SolverStats *stats = &outcomes[variant_idx].stats;
      SolverStats *game_total = &game_stats[variant->solver];
      game_total->moves += stats->moves;
      game_total->incomplete += stats->incomplete;
      game_total->no_move += stats->no_move;
      game_total->seconds += stats->seconds;

      // Result from the solver's seat, compared with static from that seat.
      const int seat = variant->solver_player;
      const int sign = seat == 0 ? 1 : -1;
      const int solved = result_sign(sign * outcomes[variant_idx].final_spread);
      const int baseline = result_sign(sign * outcomes[0].final_spread);
      improved[variant->solver][seat] += solved > baseline;
      worsened[variant->solver][seat] += solved < baseline;
    }
    for (int solver_idx = 0; solver_idx < NUM_SOLVERS; solver_idx++) {
      totals[solver_idx].moves += game_stats[solver_idx].moves;
      totals[solver_idx].incomplete += game_stats[solver_idx].incomplete;
      totals[solver_idx].no_move += game_stats[solver_idx].no_move;
      totals[solver_idx].seconds += game_stats[solver_idx].seconds;
    }

    printf("%-6d %-6d %+-9d %+-9d %+-9d %+-9d %+-9d %-8d %-8d %d\n",
           game_index + 1, seed, outcomes[0].final_spread,
           outcomes[1].final_spread, outcomes[2].final_spread,
           outcomes[3].final_spread, outcomes[4].final_spread,
           game_stats[SOLVER_ANTISTATIC].incomplete,
           game_stats[SOLVER_NORMAL].incomplete,
           game_stats[SOLVER_ANTISTATIC].no_move +
               game_stats[SOLVER_NORMAL].no_move);
    // Flush each row so a run that is stopped keeps every finished game.
    (void)fflush(stdout);

    char *path = get_formatted_string("%s/game_%04d_seed_%d.txt", out_dir,
                                      game_index + 1, seed);
    write_string_to_file(path, "w", string_builder_peek(log), error_stack);
    assert(error_stack_is_empty(error_stack));
    free(path);
    string_builder_destroy(log);
  }

  printf("\nSpreads above are player one's final spread.\n");
  printf("Endgames played: %d\n", games_played);
  for (int solver_idx = 0; solver_idx < NUM_SOLVERS; solver_idx++) {
    for (int seat = 0; seat < 2; seat++) {
      printf("%s as P%d: better result than static in %d, worse in %d\n",
             solver_names[solver_idx], seat + 1, improved[solver_idx][seat],
             worsened[solver_idx][seat]);
    }
  }
  for (int solver_idx = 0; solver_idx < NUM_SOLVERS; solver_idx++) {
    printf("%s solver: %d moves, %.2fs total, %d incomplete, %d no move\n",
           solver_names[solver_idx], totals[solver_idx].moves,
           totals[solver_idx].seconds, totals[solver_idx].incomplete,
           totals[solver_idx].no_move);
  }
  printf("Game logs: %s/\n", out_dir);

  error_stack_destroy(error_stack);
  for (int solver_idx = 0; solver_idx < NUM_SOLVERS; solver_idx++) {
    endgame_ctx_destroy(ctxs[solver_idx]);
  }
  endgame_results_destroy(results);
  move_list_destroy(move_list);
  config_destroy(config);
}
