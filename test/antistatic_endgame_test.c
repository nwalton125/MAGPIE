// Antistatic endgame experiment: how much does solving endgames against a
// static opponent gain over playing them statically?
//
// For each seed, both players play their top static equity move until the bag
// is empty. The endgame is then played out three ways from that same
// position: static vs static (the baseline), the antistatic solver as player
// one against a static player two, and the reverse. Everything before the
// endgame is shared, so any difference in the result comes from the endgame.
//
// The antistatic player runs the endgame solver on each of its turns with
// opponent_static set: its opponent is modeled exactly as the static player it
// faces.
//
// Environment variables (all optional):
//   ANTISTATIC_GAMES     number of games (default 10)
//   ANTISTATIC_SEED      seed of the first game; game i uses seed + i
//                        (default 1)
//   ANTISTATIC_SECONDS   per-move endgame solver time limit (default 10)
//   ANTISTATIC_FIRSTWIN  1 = first-win search (default), 0 = maximize spread
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
#include "../src/ent/rack.h"
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
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

enum {
  ANTISTATIC_NONE = -1,
  // Variants: 0 = static vs static, 1 = antistatic player one,
  // 2 = antistatic player two.
  NUM_VARIANTS = 3,
};

typedef struct EndgameOutcome {
  int final_spread; // player one's score minus player two's
  int antistatic_moves;
  int antistatic_timeouts;
  double antistatic_seconds;
} EndgameOutcome;

static int env_int(const char *name, int default_value) {
  const char *value = getenv(name);
  return value ? atoi(value) : default_value;
}

static double env_double(const char *name, double default_value) {
  const char *value = getenv(name);
  return value ? atof(value) : default_value;
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
static void log_and_play_move(StringBuilder *log, const Move *move,
                              Game *game, const char *annotation) {
  const LetterDistribution *ld = game_get_ld(game);
  const int on_turn = game_get_player_on_turn_index(game);
  string_builder_add_formatted_string(log, "  P%d ", on_turn + 1);
  string_builder_add_rack(
      log, player_get_rack(game_get_player(game, on_turn)), ld, false);
  string_builder_add_string(log, "  ");
  string_builder_add_move(log, game_get_board(game), move, ld, true);
  play_move(move, game, NULL);
  string_builder_add_formatted_string(
      log, "  %d-%d%s\n",
      equity_to_int(player_get_score(game_get_player(game, 0))),
      equity_to_int(player_get_score(game_get_player(game, 1))), annotation);
}

// Picks the antistatic player's move with the endgame solver. Returns false
// if the solver produced no move.
static bool choose_antistatic_move(Game *game, EndgameCtx **ctx,
                                   EndgameResults *results, double seconds,
                                   bool first_win, Move *out_move,
                                   bool *timed_out, double *elapsed) {
  ThreadControl *thread_control = thread_control_create();
  thread_control_set_status(thread_control, THREAD_CONTROL_STATUS_STARTED);
  EndgameArgs args = {0};
  args.thread_control = thread_control;
  args.game = game;
  args.plies = MAX_VARIANT_LENGTH;
  args.tt_fraction_of_mem = 0.05;
  args.initial_small_move_arena_size = DEFAULT_INITIAL_SMALL_MOVE_ARENA_SIZE;
  args.num_threads = 1;
  args.forced_pass_bypass = true;
  args.num_top_moves = 1;
  args.soft_time_limit = seconds;
  args.hard_time_limit = seconds;
  args.external_deadline_ns =
      ctimer_monotonic_ns() + (int64_t)(seconds * 1.0e9);
  args.seed = 42;
  args.first_win = first_win;
  args.opponent_static = true;

  Timer timer;
  ctimer_start(&timer);
  ErrorStack *error_stack = error_stack_create();
  endgame_solve(ctx, &args, results, error_stack);
  *elapsed = ctimer_elapsed_seconds(&timer);
  *timed_out = *elapsed >= seconds * 0.99;
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
  return true;
}

// Plays out the endgame from `start`, with `antistatic_player` (or nobody)
// using the solver and everyone else playing statically.
static EndgameOutcome play_out_endgame(const Game *start, int antistatic_player,
                                       double seconds, bool first_win,
                                       MoveList *move_list, EndgameCtx **ctx,
                                       EndgameResults *results,
                                       StringBuilder *log) {
  EndgameOutcome outcome = {0};
  Game *game = game_duplicate(start);
  Move move;
  while (game_get_game_end_reason(game) == GAME_END_REASON_NONE) {
    if (game_get_player_on_turn_index(game) == antistatic_player) {
      bool timed_out = false;
      double elapsed = 0.0;
      const bool ok = choose_antistatic_move(game, ctx, results, seconds,
                                             first_win, &move, &timed_out,
                                             &elapsed);
      assert(ok);
      outcome.antistatic_moves++;
      outcome.antistatic_timeouts += timed_out;
      outcome.antistatic_seconds += elapsed;
      char annotation[64];
      snprintf(annotation, sizeof(annotation), "  [solver %.2fs%s]", elapsed,
               timed_out ? ", TIMED OUT" : "");
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
  char *empty_cgp_cmd =
      get_formatted_string("cgp %s -lex %s;", EMPTY_CGP_WITHOUT_OPTIONS,
                           lexicon);
  load_and_exec_config_or_die(config, empty_cgp_cmd);
  free(empty_cgp_cmd);
  Game *game = config_get_game(config);
  MoveList *move_list = move_list_create(1);
  EndgameResults *results = endgame_results_create();
  EndgameCtx *ctx = NULL;

  printf("antistatic endgame experiment: %d games from seed %d, %.1fs per "
         "move, %s, lexicon %s\n",
         num_games, first_seed, seconds,
         first_win ? "first-win" : "max spread", lexicon);
  printf("%-6s %-6s %-9s %-9s %-9s %-7s\n", "game", "seed", "static",
         "anti-P1", "anti-P2", "timeouts");

  // Outcome changes relative to the baseline, from the antistatic player's
  // point of view: [0] = antistatic player one, [1] = antistatic player two.
  int improved[2] = {0};
  int worsened[2] = {0};
  long spread_gain[2] = {0};
  int total_timeouts = 0;
  int total_antistatic_moves = 0;
  double total_antistatic_seconds = 0.0;
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
    const char *variant_names[NUM_VARIANTS] = {
        "Static vs static", "Antistatic P1 vs static P2",
        "Static P1 vs antistatic P2"};
    const int antistatic_players[NUM_VARIANTS] = {ANTISTATIC_NONE, 0, 1};
    for (int variant = 0; variant < NUM_VARIANTS; variant++) {
      string_builder_add_formatted_string(log, "\n%s:\n",
                                          variant_names[variant]);
      outcomes[variant] = play_out_endgame(
          game, antistatic_players[variant], seconds, first_win, move_list,
          &ctx, results, log);
      total_timeouts += outcomes[variant].antistatic_timeouts;
      total_antistatic_moves += outcomes[variant].antistatic_moves;
      total_antistatic_seconds += outcomes[variant].antistatic_seconds;
    }

    const int baseline = outcomes[0].final_spread;
    for (int player = 0; player < 2; player++) {
      // Spreads from the antistatic player's point of view.
      const int sign = player == 0 ? 1 : -1;
      const int antistatic = sign * outcomes[player + 1].final_spread;
      const int static_result = sign * baseline;
      improved[player] += result_sign(antistatic) > result_sign(static_result);
      worsened[player] += result_sign(antistatic) < result_sign(static_result);
      spread_gain[player] += antistatic - static_result;
    }
    printf("%-6d %-6d %+-9d %+-9d %+-9d %d\n", game_index + 1, seed, baseline,
           outcomes[1].final_spread, outcomes[2].final_spread,
           outcomes[1].antistatic_timeouts + outcomes[2].antistatic_timeouts);

    char *path =
        get_formatted_string("%s/game_%04d_seed_%d.txt", out_dir,
                             game_index + 1, seed);
    FILE *file = fopen_or_die(path, "w");
    fputs(string_builder_peek(log), file);
    fclose(file);
    free(path);
    string_builder_destroy(log);
  }

  printf("\nSpreads above are player one's final spread.\n");
  printf("Endgames played: %d\n", games_played);
  for (int player = 0; player < 2; player++) {
    printf("Antistatic as P%d: better result in %d, worse in %d, mean spread "
           "gain %+.2f\n",
           player + 1, improved[player], worsened[player],
           games_played > 0 ? (double)spread_gain[player] / games_played
                            : 0.0);
  }
  printf("Antistatic solver: %d moves, %.2fs total, %d timed out\n",
         total_antistatic_moves, total_antistatic_seconds, total_timeouts);
  printf("Game logs: %s/\n", out_dir);

  endgame_ctx_destroy(ctx);
  endgame_results_destroy(results);
  move_list_destroy(move_list);
  config_destroy(config);
}
